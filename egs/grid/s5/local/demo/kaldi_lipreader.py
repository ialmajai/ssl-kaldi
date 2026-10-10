#!/usr/bin/env python3
# Copyright   2026  (author: Ibrahim Almajai)
# Apache 2.0
"""
Interactive lipreading (VSR) demo, Kaldi decode pipeline.

Reuses this recipe's own feature-extraction code (local/compute_avhubert_feats.py)
so the demo sees exactly the same mean-face ROI + AV-HuBERT features training did
-- rather than icefall's Conformer-CTC / k2 decode, this shells out to a
Kaldi chain model trained on this recipe's features (run_demo_base_lor10_aug2x.sh).

    video -> dlib 68-landmarks + mean-face alignment  (compute_avhubert_feats)
          -> mouth ROI crop, 88x88 grey
          -> AV-HuBERT feature extract (LoRA-32 base model, layer 10)
          -> write to a scratch Kaldi data dir
          -> local/demo/kaldi_decode_e2e.sh (upsample x2 -> flat-start e2e
             chain decode -> MBR word confidences)
          -> fused MBR + TruCLeS word confidence (local/demo/trucles/score.py)
"""

from __future__ import annotations

import argparse
import logging
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
from kaldiio import WriteHelper

RECIPE_ROOT = Path(__file__).resolve().parents[2]  # egs/grid/s5

# Bundled py3.8-compatible MediaPipe (this env has no mediapipe of its own,
# and a plain pip install would likely drag in a protobuf version
# incompatible with fairseq -- vendored from icefall's already-working
# bundle rather than fighting that conflict, see download/README.md if one
# exists, or just: it's a straight copy). Must be imported BEFORE anything
# that pulls in this env's own protobuf 5.x (fairseq, tensorboard -- both
# imported lazily inside load_avhubert() below): whichever protobuf loads
# first wins. Missing/broken degrades to no fallback, never a crash.
MEDIAPIPE_DIR = RECIPE_ROOT / "download/mediapipe-py38"
try:
    sys.path.insert(0, str(MEDIAPIPE_DIR))
    import mediapipe as _mediapipe  # noqa: E402
except Exception:
    _mediapipe = None

sys.path.insert(0, str(RECIPE_ROOT / "local"))
from compute_avhubert_feats import (  # noqa: E402
    load_avhubert,
    load_meanface_aligner,
    meanface_tracking,
    extract_avhubert,
)
from persistent_decoder import CHAIN_DIR, PersistentE2EDecoder, apply_cmvn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _rss_mb() -> float:
    """This process's peak resident set size so far, in MB (Linux; ru_maxrss
    is KB there, unlike macOS's bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _gpu_mb() -> float | None:
    """This process's own CUDA allocation, in MB (None without a GPU)."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / 1024**2


def _perf_log(msg: str) -> None:
    """Timing/memory diagnostics, printed directly to stderr rather than via
    the `logging` module: something in the fairseq/hydra import chain that
    load_avhubert() pulls in silences logger.info output for the rest of the
    process after it runs (confirmed reproducibly -- lines logged before
    load_avhubert() show up, identical lines logged after it don't, with no
    exception and no change to the logger's configured level/handlers).
    Root cause not chased further since it doesn't affect the demo's actual
    behavior, only these diagnostics; print() is unaffected either way."""
    print(f"[perf] {msg}", file=sys.stderr, flush=True)


def _tree_rss_kb(root_pid: int) -> int:
    """Combined RSS (KB) of root_pid and every descendant, at any depth --
    kaldi_decode.sh's real work happens several forks down (bash ->
    shared/*.sh -> run.pl (Perl) -> the actual Kaldi binary), so only
    checking direct children badly undercounts. One `ps -eo` snapshot,
    walked as a tree, rather than N per-level `ps --ppid` calls."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,rss="], capture_output=True, text=True,
        ).stdout
    except Exception:
        return 0
    children: dict[int, list[int]] = {}
    rss_by_pid: dict[int, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        pid, ppid, rss = (int(x) for x in parts)
        rss_by_pid[pid] = rss
        children.setdefault(ppid, []).append(pid)
    total = 0
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        total += rss_by_pid.get(pid, 0)
        stack.extend(children.get(pid, []))
    return total


def _run_with_peak_rss(cmd, cwd):
    """Like subprocess.run(capture_output=True, text=True), plus the peak
    combined RSS (MB) of the whole process tree it spawns -- kaldi_decode.sh
    execs many short-lived Kaldi binaries as its own descendants, none of
    which show up in this (Python) process's own resource.getrusage().
    Polling `ps`, not /usr/bin/time -v, because the latter only sees its
    direct child (the bash script), not the deeper binaries it execs."""
    import threading

    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    peak_kb = [0]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak_kb[0] = max(peak_kb[0], _tree_rss_kb(proc.pid))
            stop.wait(0.05)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    stdout, stderr = proc.communicate()
    stop.set()
    t.join()
    return proc.returncode, stdout, stderr, peak_kb[0] / 1024

MIN_FRAMES = 30  # below training's 74 (make_avhubert.sh) so short webcam clips still decode
LAYER = 10
CKPT = RECIPE_ROOT / "input/avhubert_base_ctc_lora.pt"  # LoRA-32 base, see NOTES.md
# Fused MBR + TruCLeS CEM word confidence (local/demo/trucles/score.py).
FUSION_DIR = RECIPE_ROOT / "exp/trucles/base_lor10_aug2x/fusion"
DLIB_PREDICTOR = RECIPE_ROOT / "input/shape_predictor_68_face_landmarks.dat"
MEAN_FACE = RECIPE_ROOT / "av_hubert/avhubert/20words_mean_face.npy"
AVHUBERT_PATH = RECIPE_ROOT / "av_hubert"
DECODE_SCRIPTS = {
    # fmllr: GMM-SAT decode_fmllr (per-utterance transform) -> fMLLR feature
    #   build -> TDNN-F chain decode. 6.72% WER on frozen base layer 4 only;
    #   the GMM decode pass dominates latency (see NOTES.md's profiling section).
    "fmllr": RECIPE_ROOT / "local/demo/kaldi_decode.sh",
    # e2e: straight nnet3 chain decode on raw (PCA-free) upsampled features,
    #   no GMM/fMLLR pass at all. 6.33% WER with this demo's encoder.
    "e2e": RECIPE_ROOT / "local/demo/kaldi_decode_e2e.sh",
}
PCA_MODEL = RECIPE_ROOT / "layer_sweep_mf/pca-30d-l4.pt"
UPSAMPLE_FACTOR = 2  # matches training: 25fps raw -> 50fps for the GMM/chain
DETECT_EVERY = 2  # attempt dlib detection every Nth frame while tracking holds;
# meanface_tracking/detect_landmarks_list revert to every frame as soon as
# dlib fails, so this only skips frames during easy, stable tracking.


def _load_mediapipe_fallback():
    """Build a fallback landmarker from MediaPipe Face Detection (BlazeFace,
    short-range): fast (~17 ms/frame CPU) and robust to glasses, tight
    framing and mild pose, exactly where dlib's HOG detector fails.

    Returns a callable (BGR frame) -> (68, 2) int array or None. Only the
    points our pipeline's STABLE_PTS actually reads are meaningful: the eye
    blocks 36:42 / 42:48 (set to real MediaPipe eye keypoints, so the
    similarity-transform fit's interocular distance stays valid). The
    unset points, including the nose tip (33, also in STABLE_PTS), fall
    back to MediaPipe's mouth-center point -- an approximation icefall's own
    production demo already ships with (same STABLE_PTS as ours), diluted
    by the temporal smoothing window before the fit runs; not attempted to
    be fixed here since this is a straight port of their already-validated
    approach. Ported from icefall's demo.py."""
    import cv2

    if _mediapipe is None:
        raise RuntimeError(f"bundled mediapipe not importable ({MEDIAPIPE_DIR})")
    mp = _mediapipe

    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )
    KP = mp.solutions.face_detection.FaceKeyPoint

    def fallback(image_bgr):
        h, w = image_bgr.shape[:2]
        result = detector.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not result.detections:
            return None
        kps = result.detections[0].location_data.relative_keypoints

        def px(k):
            return (kps[k].x * w, kps[k].y * h)

        lm = np.full((68, 2), px(KP.MOUTH_CENTER), dtype=float)
        lm[36:42] = px(KP.RIGHT_EYE)  # subject's right eye = dlib 36-41
        lm[42:48] = px(KP.LEFT_EYE)   # subject's left eye = dlib 42-47
        return np.round(lm).astype(int)

    return fallback


def _degrade_roi(roi: np.ndarray, noise_sigma: float, blur_sigma: float) -> np.ndarray:
    """Optionally blur then add Gaussian noise to ROI frames (T,H,W uint8),
    for the UI's robustness playground. Blur first: camera blur precedes
    sensor noise in a real pipeline. (Same as icefall's demo.py.)"""
    import cv2

    out = roi.astype(np.float32)
    if blur_sigma > 0:
        out = np.stack([cv2.GaussianBlur(f, (0, 0), blur_sigma) for f in out])
    if noise_sigma > 0:
        rng = np.random.default_rng()
        out = out + rng.normal(0.0, noise_sigma, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


class KaldiLipReader:
    """Loads dlib + AV-HuBERT once; each call to recognize() runs one clip
    through feature extraction (in-process) then the Kaldi decode pipeline
    (subprocess, since it's a multi-stage Kaldi shell pipeline)."""

    def __init__(self, device: torch.device | None = None, fp16: bool = False,
                idle_offload_seconds: int = 300):
        import dlib

        t0 = time.time()
        self.fp16 = fp16
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        if not DLIB_PREDICTOR.is_file():
            raise FileNotFoundError(f"dlib predictor not found: {DLIB_PREDICTOR}")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(DLIB_PREDICTOR))

        # Robust per-frame fallback, tried only when the fast HOG detector
        # fails (glasses, tight framing, mild pose). Missing/broken
        # MediaPipe degrades to the old behaviour (no fallback), never a crash.
        self.fallback_landmarker = None
        try:
            self.fallback_landmarker = _load_mediapipe_fallback()
            logger.info("MediaPipe fallback landmarker enabled")
        except Exception:
            logger.exception("MediaPipe fallback unavailable; continuing without")

        args_ns = argparse.Namespace(
            path=str(AVHUBERT_PATH), ckpt=str(CKPT), layer=LAYER,
            mean_face=str(MEAN_FACE),
        )
        self.model, self.transform = load_avhubert(args_ns, self.device)
        if self.fp16:
            self.model = self.model.half()
        self.aligner, self.mean_face = load_meanface_aligner(args_ns)
        _perf_log(f"Loaded AV-HuBERT base model (layer {LAYER}) from {CKPT}"
                 f"{' [fp16]' if self.fp16 else ''}")

        # PCA + upsampling done in-process (see _pca_and_upsample below)
        # instead of shelling out to shared/make_pca_features.sh +
        # shared/upsample_features.sh: those are batch scripts built for
        # thousands of utterances, and for a single ~75-frame clip their
        # fixed overhead (process spawns, split_scp, a fresh Python
        # interpreter + torch import per stage) was ~4.85s out of a ~9.9s
        # total decode latency for math that takes microseconds. Same
        # weights, loaded once here instead of once per request from disk.
        pca_state = torch.load(str(PCA_MODEL), map_location="cpu", weights_only=True)
        self.pca_components = pca_state["components"].float()
        self.pca_mean = pca_state["mean"].float()

        # Persistent e2e-chain decoder: nnet3-latgen-faster loaded once,
        # kept running across requests instead of reloaded per request --
        # see persistent_decoder.py's module docstring for how and why
        # (measured ~4-6x faster decode this way). e2e route only; the
        # fmllr route still shells out per request via kaldi_decode.sh.
        self._persistent_decoder = PersistentE2EDecoder()
        self._last_utt_id = None  # for refine_confidence(); set by recognize()
        self._last_feats = None   # (upsampled, CMVN) features, for the fused confidence
        try:
            from trucles.score import FusedConfidence
            self._fused = FusedConfidence(FUSION_DIR, CHAIN_DIR, RECIPE_ROOT)
        except Exception:
            logger.exception("Fused confidence unavailable; using Kaldi calibration")
            self._fused = None

        gpu = _gpu_mb()
        _perf_log(
            f"Startup done in {time.time() - t0:.1f}s "
            f"(process RSS {_rss_mb():.0f} MB"
            + (f", GPU {gpu:.0f} MB" if gpu is not None else "") + ")"
        )

        # Idle GPU offload, ported from icefall's demo.py: on a rarely-used
        # demo, keeping AV-HuBERT resident on the GPU wastes memory other
        # jobs on this box could use. After `idle_offload_seconds` with no
        # request, move it to CPU and free the CUDA cache; the next request
        # moves it back (<1s). Only AV-HuBERT needs this -- the Kaldi decode
        # pipeline (PCA, upsample, GMM/chain decode) already runs entirely on
        # CPU via subprocess, so there's nothing else GPU-resident to offload
        # (unlike icefall's Conformer+HLG, which also live on the GPU).
        # A single lock serialises this against inference so a request can
        # never see a half-migrated model. Disabled (`<= 0`) or with no CUDA
        # device, this is a no-op.
        self._gpu_device = self.device
        self._cpu_device = torch.device("cpu")
        self._device_lock = threading.RLock()
        self._offloaded = False
        self._last_active = time.monotonic()
        self._offload_seconds = idle_offload_seconds
        if self._offload_seconds > 0 and self._gpu_device.type == "cuda":
            t = threading.Thread(target=self._idle_offload_loop, daemon=True)
            t.start()
            _perf_log(
                f"Idle GPU offload enabled: AV-HuBERT moves to CPU after "
                f"{self._offload_seconds}s idle"
            )

    def _activate(self) -> None:
        """Ensure the model is on the GPU before an inference. Caller holds
        the lock. Also stamps activity so the idle timer restarts."""
        if self._offloaded:
            _perf_log("Request arrived; restoring AV-HuBERT to GPU")
            self.model = self.model.to(self._gpu_device)
            self.device = self._gpu_device
            self._offloaded = False
        self._last_active = time.monotonic()

    def _idle_offload_loop(self) -> None:
        """Background daemon: offload the model to CPU once the demo has
        been idle for `_offload_seconds`. Re-checks under the lock so it
        never offloads a model a request just reactivated. Uses _perf_log,
        not `logger` -- logger.info() silently produces no output once
        load_avhubert() has run (see _perf_log's own docstring), and this
        loop runs for the process's whole lifetime after that point."""
        while True:
            time.sleep(min(30, self._offload_seconds))
            with self._device_lock:
                if self._offloaded:
                    continue
                idle = time.monotonic() - self._last_active
                if idle >= self._offload_seconds:
                    _perf_log(f"Idle for {idle:.0f}s; offloading AV-HuBERT to CPU")
                    self.model = self.model.to(self._cpu_device)
                    self.device = self._cpu_device
                    self._offloaded = True
                    torch.cuda.empty_cache()

    def _extract_roi_and_features(self, video_path: str, noise_sigma: float = 0.0,
                                  blur_sigma: float = 0.0):
        """(roi frames (T,88,88) uint8, AV-HuBERT features (T,768) float32)."""
        # Cache ROIs only for clips outside the temp dir: user uploads and
        # webcam recordings live there and must not leave mouth crops behind.
        tmp_root = Path(tempfile.gettempdir()).resolve()
        user_clip = tmp_root in Path(video_path).resolve().parents
        roi_frames, fps = meanface_tracking(
            video_path, self.detector, self.predictor, self.aligner, self.mean_face,
            fallback_landmarker=self.fallback_landmarker, detect_every=DETECT_EVERY,
            cache=not user_clip,
        )
        if len(roi_frames) < MIN_FRAMES:
            raise RuntimeError(
                "no face/mouth could be tracked reliably in this clip "
                f"(got {len(roi_frames)} usable frames, need >= {MIN_FRAMES}). "
                "Please use a frontal talking-face video."
            )
        if noise_sigma or blur_sigma:  # robustness playground
            roi_frames = _degrade_roi(roi_frames, noise_sigma, blur_sigma)
        transformed = self.transform(roi_frames)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(transformed))
            .unsqueeze(0).unsqueeze(0).to(self.device)
        )
        tensor = tensor.half() if self.fp16 else tensor.float()
        feats = extract_avhubert(tensor, self.model, LAYER)
        return roi_frames, feats.astype(np.float32)  # ark/PCA math stays fp32 either way

    def _pca_and_upsample(self, feats: np.ndarray) -> np.ndarray:
        """768-dim raw AV-HuBERT features (T, 768) -> 30-dim PCA'd,
        50fps-upsampled features (2T, 30) -- exactly what
        shared/make_pca_features.sh + shared/upsample_features.sh
        (--interp-mode nearest --upsample-factor 2) produce, just in-process.
        PCA: shared/pca.py's apply_pca, `(feats - mean) @ components.T`.
        Upsample: torch.nn.functional.interpolate(..., mode='nearest') with
        an integer scale factor duplicates each frame in place, i.e.
        np.repeat -- confirmed against shared/interpolate.py."""
        feats_t = torch.from_numpy(feats).float()
        pca_feats = ((feats_t - self.pca_mean) @ self.pca_components.T).numpy()
        return np.repeat(pca_feats, UPSAMPLE_FACTOR, axis=0)

    @staticmethod
    def _upsample_only(feats: np.ndarray) -> np.ndarray:
        """768-dim raw AV-HuBERT features (T, 768) -> 50fps-upsampled (2T,
        768), no PCA -- what the e2e chain (exp/chain/e2e_tdnnf_lor10norelu_cmvn_aug2x)
        was trained on. Same np.repeat as _pca_and_upsample, just skipping
        the PCA step the fMLLR route needs and this one doesn't."""
        return np.repeat(feats, UPSAMPLE_FACTOR, axis=0)

    @staticmethod
    def _write_kaldi_dir(utt_id: str, feats: np.ndarray) -> str:
        """Write a one-utterance raw-feature Kaldi data dir; caller deletes it."""
        tmpdir = tempfile.mkdtemp(prefix="vsr-kaldi-")
        with WriteHelper(f"ark,scp:{tmpdir}/feats.ark,{tmpdir}/feats.scp") as writer:
            writer(utt_id, feats)
        (Path(tmpdir) / "utt2spk").write_text(f"{utt_id} {utt_id}\n")
        (Path(tmpdir) / "spk2utt").write_text(f"{utt_id} {utt_id}\n")
        return tmpdir

    def recognize(self, video_path: str, noise_sigma: float = 0.0,
                 blur_sigma: float = 0.0, method: str = "e2e"):
        """Return (recognized_text, roi_frames (T,88,88) uint8, word_confs),
        word_confs a list of (word, confidence in [0, 1]) pairs from Kaldi's
        lattice posteriors (MBR-decoded), in sentence order.

        method: "e2e" (default -- 6.33% WER, no GMM/fMLLR pass, the
        flat-start chain decoding raw upsampled features directly, roughly
        2-3x faster) or "fmllr" (GMM-SAT decode_fmllr for a per-utterance
        transform, then the fMLLR TDNN-F chain; needs frozen base layer 4
        features, so it raises with this demo's encoder)."""
        if method not in DECODE_SCRIPTS:
            raise ValueError(f"method must be one of {list(DECODE_SCRIPTS)}, got {method!r}")
        if method == "fmllr" and (LAYER, CKPT.name) != (4, "base_vox_iter5.pt"):
            raise ValueError("fmllr route was trained on frozen base layer 4 features; "
                             f"this demo extracts layer {LAYER} from {CKPT.name}")

        # Hold the device lock across the whole request so the idle monitor
        # cannot offload AV-HuBERT mid-request; _activate() restores it to
        # the GPU first if it was offloaded, and refreshes the idle timer.
        with self._device_lock:
            self._activate()
            try:
                t0 = time.time()
                roi_frames, feats = self._extract_roi_and_features(
                    video_path, noise_sigma, blur_sigma)
                t_extract = time.time() - t0
                gpu = _gpu_mb()
                _perf_log(
                    f"ROI+AV-HuBERT extraction: {t_extract:.2f}s ({len(roi_frames)} frames"
                    + (f", GPU {gpu:.0f} MB" if gpu is not None else "") + ")"
                )

                t0 = time.time()
                feats = (self._pca_and_upsample(feats) if method == "fmllr"
                         else self._upsample_only(feats))
                _perf_log(f"{'PCA+' if method == 'fmllr' else ''}upsample (in-process): "
                          f"{time.time() - t0:.3f}s")

                if method == "e2e":
                    # Persistent decoder: no subprocess spawn, no model
                    # reload -- see persistent_decoder.py.
                    t0 = time.time()
                    cmvn_feats = apply_cmvn(feats)
                    utt_id = self._persistent_decoder.decode(cmvn_feats)
                    word_confs = self._persistent_decoder.raw_confidence(utt_id)
                    self._last_utt_id = utt_id
                    self._last_feats = (feats, cmvn_feats)
                    t_decode = time.time() - t0
                    _perf_log(
                        f"Kaldi decode pipeline (e2e, persistent): {t_decode:.2f}s "
                        f"(total recognize() latency: {t_extract + t_decode:.2f}s, "
                        f"Python process RSS {_rss_mb():.0f} MB)"
                    )
                else:
                    t0 = time.time()
                    utt_id = "demo"
                    tmpdir = self._write_kaldi_dir(utt_id, feats)
                    try:
                        returncode, stdout, stderr, peak_rss_mb = _run_with_peak_rss(
                            [str(DECODE_SCRIPTS[method]), utt_id, tmpdir], cwd=str(RECIPE_ROOT))
                        if returncode != 0:
                            logger.error(stderr[-4000:])
                            raise RuntimeError("Kaldi decode pipeline failed; see server log.")
                        for line in stderr.splitlines():
                            if line.startswith("[stage]"):
                                _perf_log(line)
                        word_confs = []
                        for line in stdout.strip().splitlines():
                            word, conf = line.rsplit(None, 1)
                            word_confs.append((word, float(conf)))
                    finally:
                        shutil.rmtree(tmpdir, ignore_errors=True)
                    t_decode = time.time() - t0
                    _perf_log(
                        f"Kaldi decode pipeline ({method}): {t_decode:.2f}s, peak process-tree RSS "
                        f"{peak_rss_mb:.0f} MB (total recognize() latency: "
                        f"{t_extract + t_decode:.2f}s, Python process RSS {_rss_mb():.0f} MB)"
                    )
            finally:
                # Restart the idle countdown from when the request finished,
                # not when it started (long clips shouldn't shorten it).
                self._last_active = time.monotonic()
        text = " ".join(w for w, _ in word_confs)
        return text, roi_frames, word_confs

    def refine_confidence(self, method: str = "e2e"):
        """Second-stage word-confidence refinement for the most recent
        recognize() call: raw MBR posteriors -> steps/conf/apply_calibration.sh's
        calibrated probabilities (logistic regression trained on real
        correct/incorrect labels, see NOTES.md's confidence-calibration
        entry) -- discriminates much better than the raw posterior, which
        scores most correct words a flat, uninformative 1.00, but costs
        ~1-2s of its own (lattice-rescale/depth/logistic-regression-eval).
        Reuses the lattice recognize() already produced on disk
        (kaldi_calibrate_conf.sh), no redecoding. Meant to be called
        *after* recognize()'s text is already shown to the user, not
        before -- see demo.py's infer(), which yields the fast raw
        confidence first and this refinement second.

        Uses the fused MBR + TruCLeS CEM confidence (trucles/score.py) when its
        model loads and the MBR and 1-best words agree; Kaldi calibration otherwise.

        e2e only: no calibration model is trained for the fmllr route.
        Returns None for method="fmllr" or if calibration fails (the
        fast raw confidence from recognize() stays on screen either way,
        so this degrading gracefully matters more than it succeeding).

        Instance method, not static: needs the persistent decoder's own
        lattice (self._persistent_decoder) and the utt-id recognize() just
        used (self._last_utt_id) -- the persistent decoder assigns a fresh
        id per request (unlike the old fixed "demo"), so both are
        per-instance state now."""
        if method != "e2e" or self._last_utt_id is None:
            return None
        t0 = time.time()
        if self._fused is not None:
            try:
                word_confs = self._fused(self._persistent_decoder.ark_path, self._last_utt_id,
                                         *self._last_feats)
                if word_confs is not None:
                    _perf_log(f"Fused MBR+CEM confidence: {time.time() - t0:.2f}s")
                    return word_confs
            except Exception:
                logger.exception("Fused confidence failed; falling back to calibration")
        script = RECIPE_ROOT / "local/demo/kaldi_calibrate_conf.sh"
        try:
            # Snapshot the persistent decoder's growing ark file to the
            # lat.1.gz/num_jobs kaldi_calibrate_conf.sh's unmodified
            # apply_calibration.sh call expects -- see
            # PersistentE2EDecoder.snapshot_for_calibration's docstring.
            self._persistent_decoder.snapshot_for_calibration()
            returncode, stdout, stderr, _ = _run_with_peak_rss(
                [str(script), self._last_utt_id], cwd=str(RECIPE_ROOT))
            if returncode != 0:
                logger.error(stderr[-4000:])
                return None
            word_confs = []
            for line in stdout.strip().splitlines():
                word, conf = line.rsplit(None, 1)
                word_confs.append((word, float(conf)))
        except Exception:
            logger.exception("Confidence calibration failed")
            return None
        _perf_log(f"Confidence calibration: {time.time() - t0:.2f}s")
        return word_confs


def _cli():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="+",
                        help="one or more video clips (e.g. grid-corpus/s1/bbaf2n.mpg). "
                             "Multiple clips run in the same process, reusing the "
                             "already-loaded model -- useful for comparing cold "
                             "(first-clip, includes model load) vs warm latency.")
    parser.add_argument("--method", choices=list(DECODE_SCRIPTS), default="e2e",
                        help="e2e (default, 6.33%% WER, no GMM/fMLLR pass, "
                             "faster) or fmllr (frozen layer-4 encoder only).")
    parser.add_argument("--fp16", action="store_true",
                        help="run AV-HuBERT in half precision (experimental).")
    parser.add_argument("--idle-offload-seconds", type=int, default=300,
                        help="Move AV-HuBERT to CPU after this many idle "
                             "seconds, restoring it on the next request "
                             "(<1s). 0 disables. Default: %(default)s")
    parser.add_argument("--sleep-between", type=float, default=0.0,
                        help="Sleep this many seconds between clips (e.g. to "
                             "exercise --idle-offload-seconds against a real "
                             "idle gap). Default: %(default)s")
    parser.add_argument("--noise", type=float, nargs="+", default=None,
                        help="Gaussian noise sigma(s) applied to the mouth "
                             "ROI before recognition (matches the UI slider, "
                             "0-50). One value per video, or one video "
                             "repeated once per value here to sweep noise "
                             "levels on a single clip.")
    parser.add_argument("--blur", type=float, nargs="+", default=None,
                        help="Gaussian blur sigma(s) applied to the mouth "
                             "ROI before recognition (matches the UI slider, "
                             "0-5). Paired index-wise with --noise; same "
                             "broadcasting rule.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Also call refine_confidence() after each "
                             "recognize(), printing the calibrated "
                             "confidences on a second line -- e2e only.")
    args = parser.parse_args()

    videos = args.video
    noise = args.noise or [0.0] * len(videos)
    blur = args.blur or [0.0] * len(videos)
    n = max(len(videos), len(noise), len(blur))
    if len(videos) == 1 and n > 1:
        videos = videos * n
    if len(noise) == 1 and n > 1:
        noise = noise * n
    if len(blur) == 1 and n > 1:
        blur = blur * n
    if not (len(videos) == len(noise) == len(blur)):
        parser.error("video/--noise/--blur must have matching lengths "
                      "(or a single value to broadcast)")

    reader = KaldiLipReader(fp16=args.fp16,
                            idle_offload_seconds=args.idle_offload_seconds)
    for i, (video, n_sigma, b_sigma) in enumerate(zip(videos, noise, blur)):
        if i > 0 and args.sleep_between > 0:
            time.sleep(args.sleep_between)
        text, _, word_confs = reader.recognize(
            video, noise_sigma=n_sigma, blur_sigma=b_sigma, method=args.method)
        print(f"[noise={n_sigma:g} blur={b_sigma:g}] {text}")
        print("raw:   " + " ".join(f"{w}({c:.2f})" for w, c in word_confs))
        if args.calibrate:
            calibrated = reader.refine_confidence(method=args.method)
            if calibrated is not None:
                print("calib: " + " ".join(f"{w}({c:.2f})" for w, c in calibrated))


if __name__ == "__main__":
    _cli()
