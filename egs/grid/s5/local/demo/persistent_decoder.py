#!/usr/bin/env python3
# Copyright   2026  (author: Ibrahim Almajai)
# Apache 2.0
"""
Persistent e2e-chain Kaldi decoder for the demo: keeps `nnet3-latgen-faster`
running across requests instead of spawning it fresh (and reloading
final.mdl from disk) for every decode -- see NOTES.md's
"Why icefall is faster" entry for the measurement that motivated this
(nnet3-latgen-faster was the single biggest contributor to our per-request
latency, and model reload was a large, avoidable part of that).

How this works, and what's still validated-but-fragile about it (each
point empirically verified against a real decode/WER before trusting it,
not assumed from Kaldi documentation):

- Kaldi's ark I/O is stream-oriented by design (this is how `cmd1 | cmd2`
  Kaldi pipelines already work): a SequentialTableReader on `ark:-` blocks
  waiting for more input rather than treating a quiet-but-open pipe as
  EOF. Confirmed by writing one utterance, NOT closing stdin, and finding
  the process still alive and correctly decoding a second utterance
  written seconds later, with the model loaded only once (no repeated
  "Removed N orphan nodes" model-load log lines).
- The `ark,scp:` output's *scp index* is only flushed at process exit/
  TableWriter destruction, not incrementally -- confirmed by writing one
  utterance to a long-lived process and finding the .scp file still empty
  seconds later even though the .ark *file* had already grown. So the scp
  index can't be polled as a "this utterance is ready" signal.
- Kaldi's per-utterance progress LOG line ("Log-like per frame for
  utterance <id> is ...", written via KALDI_LOG -> std::cerr) *is*
  delivered promptly through the stderr pipe -- confirmed the line for
  utterance N arrives ~0.1-0.2s after writing it, every time, across
  several sequential requests to the same process. Used here as the
  completion signal in place of the (deferred) scp index.
- CMVN (`apply-cmvn --norm-means=true --norm-vars=true`) for this
  per-utterance-as-its-own-speaker setup is exactly
  `(feats - feats.mean(axis=0)) / feats.std(axis=0)` (population
  variance) -- verified bit-exact (max abs diff ~5e-7, float32 rounding)
  against a real `compute-cmvn-stats` + `apply-cmvn` run before trusting
  it. Needed because the persistent process can't run a live shell pipe
  the way a fresh `steps/nnet3/decode.sh` invocation does.

What did NOT change: the actual decode flags below are copied verbatim
from `steps/nnet3/decode.sh` (defaults: beam=15.0, max-active=7000,
min-active=200, extra-left-context=0, extra-right-context=0; overrides
matching kaldi_decode_e2e.sh: frame-subsampling-factor=2,
frames-per-chunk=150, extra-left/right-context-initial/final=0,
lattice-beam=2.0, acoustic-scale=1.0) -- this is the same decode, just not
re-launched from scratch each time. post-decode-acwt=10.0 is still applied
as a separate `lattice-scale` step per request, same as decode.sh does
for post_decode_acwt != 1.0.

Confidence extraction (both the fast raw path and calibration) is left
completely unchanged: `kaldi_decode_e2e.sh`'s raw lattice-to-ctm-conf
command and `kaldi_calibrate_conf.sh` both already filter their output by
utt-id, so pointing them at the growing multi-utterance ark file this
class produces (or a gzipped snapshot of it, for calibration -- see
`snapshot_for_calibration`) works without changing either script.
"""
from __future__ import annotations

import atexit
import os
import queue
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

RECIPE_ROOT = Path(__file__).resolve().parents[2]
CHAIN_DIR = RECIPE_ROOT / "exp/chain/e2e_tdnnf_lor10norelu_cmvn_aug2x"
GRAPH_DIR = RECIPE_ROOT / "exp/chain/e2e_tree_lor10_aug2x/graph"

# Every other Kaldi-binary call in this recipe goes through a shell script
# that sources path.sh first, which is what actually puts $KALDI_ROOT/src/
# */bin on PATH (see path.sh: `export PATH=...$KALDI_ROOT/tools/...:$PWD:
# $PATH` then `. $KALDI_ROOT/tools/config/common_path.sh`, which adds
# every src/*bin dir). This module calls nnet3-latgen-faster directly from
# Python with no such shell in between, so PATH as inherited from whatever
# invoked Python (which may never have sourced path.sh) isn't reliable --
# hit this directly as a real FileNotFoundError. Resolve KALDI_ROOT the
# same env-overridable way path.sh does and reference binaries by their
# full path instead of depending on PATH.
KALDI_ROOT = Path(os.environ.get("KALDI_ROOT", "/data/git/kaldi"))
NNET3_LATGEN_FASTER = KALDI_ROOT / "src/nnet3bin/nnet3-latgen-faster"
LATTICE_SCALE = KALDI_ROOT / "src/latbin/lattice-scale"
LATTICE_TO_CTM_CONF = KALDI_ROOT / "src/latbin/lattice-to-ctm-conf"
POST_DECODE_ACWT = 10.0  # applied as a separate lattice-scale step, matching
                          # decode.sh's handling of post_decode_acwt != 1.0
# Requests per persistent-process lifetime before rotating to a fresh one:
# the output ark file only ever grows (Kaldi's ark format has no in-place
# delete), so this bounds its size on a long-running demo server rather
# than leaving genuinely unbounded growth.
ROTATE_AFTER = 200

_COMPLETION_RE = re.compile(r"Log-like per frame for utterance (\S+) is")
_FAILED_RE = re.compile(r"Did not successfully decode utterance (\S+)")


def default_decode_dir() -> Path:
    """Same ${KALDI_DEMO_SCRATCH:-$chain}/decode_demo resolution the shell
    scripts (kaldi_decode_e2e.sh, kaldi_calibrate_conf.sh) use -- so the
    persistent decoder's own ark file and the calibration snapshot both
    land in whatever writable location bare metal or Docker already
    provides (the Docker tmpfs is mounted at exactly this nested path
    inside the read-only repo mount, see demo_docker/run.sh), without
    needing any new mount."""
    import os
    scratch = os.environ.get("KALDI_DEMO_SCRATCH") or str(CHAIN_DIR)
    return Path(scratch) / "decode_demo"


class PersistentE2EDecoder:
    """One long-lived `nnet3-latgen-faster` process, fed utterances over
    its own stdin, with per-utterance completion signalled via a stderr
    log line instead of process exit. Not thread-safe for concurrent
    decode() calls -- matches this demo's existing single-request-at-a-time
    Gradio queue (see demo.py's demo.queue(default_concurrency_limit=1))."""

    def __init__(self, scratch_dir: Path | None = None):
        self.scratch_dir = Path(scratch_dir) if scratch_dir else (default_decode_dir() / "persistent")
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        self._lock = threading.Lock()
        self._pending: dict[str, "queue.Queue[str | None]"] = {}
        self._start_process()
        # subprocess.Popen children are not killed automatically when the
        # parent exits -- without this, every CLI invocation (this demo's
        # own heavily-used test path, `python kaldi_lipreader.py <clips>`)
        # would leave an orphaned nnet3-latgen-faster sitting forever,
        # blocked on a stdin read that will never come. Only covers normal
        # interpreter exit (SystemExit, falling off the end of main, ...),
        # not a hard `kill -9` on a long-running demo server -- accepted,
        # not worth complicating the server's own signal handling for.
        atexit.register(self.shutdown)

    def _start_process(self) -> None:
        self.ark_path = self.scratch_dir / "persist_lat.ark"
        self.scp_path = self.scratch_dir / "persist_lat.scp"
        for p in (self.ark_path, self.scp_path):
            p.unlink(missing_ok=True)
        cmd = [
            str(NNET3_LATGEN_FASTER),
            "--frame-subsampling-factor=2",
            "--frames-per-chunk=150",
            "--extra-left-context=0",
            "--extra-right-context=0",
            "--extra-left-context-initial=0",
            "--extra-right-context-final=0",
            "--minimize=false",
            "--max-active=7000",
            "--min-active=200",
            "--beam=15.0",
            "--lattice-beam=2.0",
            "--acoustic-scale=1.0",
            "--allow-partial=true",
            f"--word-symbol-table={GRAPH_DIR / 'words.txt'}",
            str(CHAIN_DIR / "final.mdl"),
            str(GRAPH_DIR / "HCLG.fst"),
            "ark:-",
            f"ark,scp:{self.ark_path},{self.scp_path}",
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(RECIPE_ROOT)
        )
        self._requests_this_process = 0
        t = threading.Thread(target=self._stderr_reader, daemon=True)
        t.start()

    def _stderr_reader(self) -> None:
        proc = self.proc  # capture: self.proc may be replaced by a rotation
        for line in proc.stderr:
            text = line.decode(errors="replace")
            m = _COMPLETION_RE.search(text) or _FAILED_RE.search(text)
            if m:
                utt = m.group(1)
                with self._lock:
                    q = self._pending.pop(utt, None)
                if q is not None:
                    q.put(text.strip())

    def decode(self, feats: np.ndarray, timeout: float = 20.0) -> str:
        """feats: (T, 768) float32, already CMVN-normalised and upsampled
        to 50fps (see apply_cmvn below). Returns the utt-id this request
        was decoded under -- the caller reads results (fast raw confidence
        or, via snapshot_for_calibration, the calibrated path) by filtering
        `self.ark_path` for that key, exactly like the old per-request
        kaldi_decode_e2e.sh did against its own fresh lat.1.gz."""
        import kaldiio.matio as matio

        with self._lock:
            if self.proc.poll() is not None:
                # Died between requests (crash, OOM, ...): restart clean
                # rather than serve every subsequent request a timeout.
                self._start_process()
            elif self._requests_this_process >= ROTATE_AFTER:
                # Rotate lazily, at the start of the request that would
                # exceed the cap -- NOT at the end of the previous request.
                # _start_process() reuses a fixed ark filename (unlinking
                # it first), so rotating right after that previous
                # request's own completion signal deleted its lattice
                # before the caller's very next line (raw_confidence() /
                # snapshot_for_calibration()) could read it. Caught via a
                # full 3985-utterance WER regression: exactly 19 utterances
                # (3985/200 ~= 19.9 rotations) came back with an empty
                # transcript. Rotating here instead means the previous
                # request's ark file is only ever replaced after its
                # caller has already finished reading it.
                old_proc = self.proc
                self._start_process()
                try:
                    old_proc.stdin.close()
                except Exception:
                    pass
            self._counter += 1
            utt_id = f"persist{self._counter}"
            q: "queue.Queue[str | None]" = queue.Queue()
            self._pending[utt_id] = q
            self._requests_this_process += 1

        self.proc.stdin.write(f"{utt_id} ".encode())
        matio.write_array(self.proc.stdin, feats.astype(np.float32))
        self.proc.stdin.flush()

        try:
            q.get(timeout=timeout)
        except queue.Empty:
            with self._lock:
                self._pending.pop(utt_id, None)
            raise TimeoutError(
                f"persistent decoder: no completion signal for {utt_id} "
                f"within {timeout}s"
            )

        return utt_id

    def snapshot_for_calibration(self, decode_dir: Path | None = None) -> None:
        """Gzip the current ark file to decode_dir/lat.1.gz and write
        decode_dir/num_jobs=1, reconstructing the file-based contract
        steps/conf/apply_calibration.sh expects (it reads $latdir/lat.JOB.gz
        and $latdir/num_jobs) -- so that script needs no changes at all.
        The snapshot contains every utterance decoded so far, not just the
        current request's, but apply_calibration.sh's own output already
        gets filtered by utt-id downstream (kaldi_calibrate_conf.sh's awk),
        exactly like the fast raw-confidence path already does against
        this same growing ark file."""
        decode_dir = Path(decode_dir) if decode_dir else default_decode_dir()
        decode_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            ark_path = self.ark_path
        subprocess.run(
            f"gzip -c {ark_path} > {decode_dir}/lat.1.gz",
            shell=True, check=True,
        )
        (decode_dir / "num_jobs").write_text("1\n")

    def raw_confidence(self, utt_id: str) -> list[tuple[str, float]]:
        """Fast raw MBR posterior confidence for one utterance -- same
        command kaldi_decode_e2e.sh's fast path already ran against a
        fresh lat.1.gz, just pointed at this growing ark file and letting
        the existing awk filter pick out `utt_id`'s line."""
        with self._lock:
            ark_path = self.ark_path
        # utt_id is always internally generated (f"persist{n}"), never
        # user input, but shlex.quote it anyway rather than rely on that.
        quoted_utt = shlex.quote(utt_id)
        cmd = (
            f"{LATTICE_SCALE} --acoustic-scale={POST_DECODE_ACWT} ark:{ark_path} ark:- 2>/dev/null | "
            f"{LATTICE_TO_CTM_CONF} --acoustic-scale=0.1 --decode-mbr=true ark:- - 2>/dev/null | "
            f"utils/int2sym.pl -f 5 data/lang/words.txt | "
            f"awk -v utt={quoted_utt} '$1==utt {{print $5, $6}}'"
        )
        out = subprocess.run(cmd, shell=True, cwd=str(RECIPE_ROOT),
                             capture_output=True, text=True, check=True).stdout
        word_confs = []
        for line in out.strip().splitlines():
            word, conf = line.rsplit(None, 1)
            word_confs.append((word, float(conf)))
        return word_confs

    def shutdown(self) -> None:
        """Close stdin (closing an open pipe's write end delivers EOF,
        Kaldi's normal, clean way to finish a `ark:-` stream) and wait
        briefly; force-kill if it doesn't exit on its own."""
        with self._lock:
            proc = self.proc
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def apply_cmvn(feats: np.ndarray) -> np.ndarray:
    """In-process equivalent of `apply-cmvn --norm-means=true
    --norm-vars=true` for this recipe's per-utterance-as-its-own-speaker
    setup: (x - mean) / population-stddev per dimension, computed over
    this utterance's own frames. See this module's docstring for the
    verification (bit-exact vs a real compute-cmvn-stats + apply-cmvn
    run, max abs diff ~5e-7)."""
    x = feats.astype(np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return (x - mean) / std
