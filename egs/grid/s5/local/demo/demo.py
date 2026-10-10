#!/usr/bin/env python3
# Copyright   2026  (author: Ibrahim Almajai)
# Apache 2.0
"""
Interactive lipreading (VSR) demo for this recipe's LoRA-32 base/layer-10 system.

Same Gradio interface as icefall/egs/grid/VSR/conformer_ctc2/demo.py --
layout, CSS, the rotating grammar-sentence ticker, the noise/blur robustness
sliders, and per-word confidence highlighting -- with the decode pipeline
swapped for this recipe's own: LoRA-32 base AV-HuBERT layer 10 -> Kaldi, decoded by
one of two interchangeable routes (KaldiLipReader.recognize's `method`
param; default "e2e"):
  - "e2e" (default): upsample -> flat-start e2e chain decode directly on raw
    768-dim features. 6.33% WER, no GMM/fMLLR pass, roughly 2-3x faster.
  - "fmllr": PCA-30 -> GMM-SAT decode_fmllr (per-utterance fMLLR transform)
    -> fMLLR TDNN-F chain. 6.72% WER, slower. Needs frozen layer-4 features,
    so it refuses to run with this demo's encoder.
...instead of icefall's Conformer-CTC / k2 lattice decoding. Word confidence
comes from Kaldi's `lattice-to-ctm-conf` (MBR lattice posteriors), then is
refined by a logistic fusion of MBR with a TruCLeS confidence model
(local/demo/trucles/), falling back to Kaldi calibration. See kaldi_lipreader.py.

Not carried over (deployment-specific to icefall's hosted public demo, not
part of the decode pipeline itself): the Cloudflare-tunnel/Docker isolation,
per-visitor quotas, out-of-domain detection/calibration, and isotonic
word-confidence recalibration.
Idle GPU offload (--idle-offload-seconds, see KaldiLipReader) *was* ported.

Usage:
    # CLI -- print the recognised text for one video:
    ./local/demo/demo.py grid-corpus/s1/bbaf2n.mpg

    # Gradio web UI:
    ./local/demo/demo.py --ui
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaldi_lipreader import KaldiLipReader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MIN_DURATION_S = 2.0
MAX_DURATION_S = 15.0
MAX_DURATION_WEBCAM_S = 20.0
DURATION_TOL_S = 0.1
TARGET_FPS = 25.0  # GRID / AV-HuBERT rate; other rates are resampled first

RECIPE_ROOT = Path(__file__).resolve().parents[2]

# Selectable back-ends, chosen by a toggle in the UI (off = Kaldi, the
# default). "kaldi" is this recipe's own pipeline (in-process); "icefall" is
# the GRID Conformer-CTC system of the icefall recipe (egs/grid/VSR, phone
# lexicon, AV-HuBERT layer 8), served by icefall's conformer_ctc2/demo_api.py
# on the internal container network.
ICEFALL_API_URL = os.environ.get("ICEFALL_API_URL", "http://vsr-icefall-api:8000")
ICEFALL_TIMEOUT_S = 180


def _recognize_icefall(clip: str, noise: float, blur: float):
    """Run one clip through the icefall service. Returns (text, roi,
    word_confs, ood) like icefall's LipReader.recognize; raises ValueError
    with a user-facing message for input problems and RuntimeError when the
    service cannot be reached."""
    import requests

    try:
        with open(clip, "rb") as f:
            r = requests.post(
                f"{ICEFALL_API_URL}/recognize",
                files={"video": (Path(clip).name, f, "application/octet-stream")},
                data={"noise": str(noise), "blur": str(blur)},
                timeout=ICEFALL_TIMEOUT_S,
            )
    except requests.RequestException as e:
        raise RuntimeError(f"icefall service unreachable: {e}") from e
    if r.status_code == 422:
        raise ValueError(r.json().get("detail", "This clip could not be processed."))
    r.raise_for_status()
    out = r.json()
    roi = np.load(io.BytesIO(base64.b64decode(out["roi"])), allow_pickle=False)
    word_confs = [(w, float(c)) for w, c in out["word_confs"]]
    return out["text"], roi, word_confs, out.get("ood")

GRID_GRAMMAR_MD = """
Every GRID sentence follows a fixed 6-word grammar:

**command · colour · preposition · letter · digit · adverb**

| slot | choices |
| --- | --- |
| command | bin, lay, place, set |
| colour | blue, green, red, white |
| preposition | at, by, in, with |
| letter | a-z (except **w**) |
| digit | zero, one, two, three, four, five, six, seven, eight, nine |
| adverb | again, now, please, soon |

Example: **place green with h eight now**
"""

ACKNOWLEDGEMENTS_MD = """
---
**Acknowledgements**: This research demo uses the
[GRID audiovisual sentence corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/)
(Cooke, Barker, Cunningham & Shao, 2006; CC BY 4.0) and Meta's
[AV-HuBERT](https://github.com/facebookresearch/av_hubert) visual front-end
(Shi et al., 2022; non-commercial research license), and is built with
[Kaldi](http://kaldi-asr.org/) and [dlib](http://dlib.net/) facial landmarks.
Interface modeled on the [icefall](https://github.com/k2-fsa/icefall) / k2
version of this demo, with Kaldi swapped in for the decode pipeline.
"""


def _grammar_fst_svg() -> str:
    """The GRID grammar drawn as a linear ("sausage") FST. Copied verbatim
    from icefall's demo.py -- self-contained, no icefall/k2 dependency."""
    slots = [
        ("command", ["bin", "lay", "place", "set"]),
        ("colour", ["blue", "green", "red", "white"]),
        ("preposition", ["at", "by", "in", "with"]),
        ("letter (no w)", ["a", "b", "…", "z"]),
        ("digit", ["zero", "one", "…", "nine"]),
        ("adverb", ["again", "now", "please", "soon"]),
    ]
    seg_w, x0, y, r = 170, 40, 170, 14
    apexes = [-84, -28, 28, 84]
    parts = []
    for i, (title, words) in enumerate(slots):
        x1, x2 = x0 + i * seg_w, x0 + (i + 1) * seg_w
        cx = (x1 + x2) // 2
        parts.append(
            f'<text x="{cx}" y="28" text-anchor="middle" font-weight="bold" '
            f'font-size="20" fill="currentColor">{title}</text>'
        )
        for a, w in zip(apexes, words):
            dash = ' stroke-dasharray="4 4"' if w == "…" else ""
            parts.append(
                f'<path d="M {x1 + r} {y} Q {cx} {y + 2 * a} {x2 - r} {y}" '
                f'fill="none" stroke="currentColor" stroke-opacity="0.45" '
                f'marker-end="url(#arr)"{dash}/>'
            )
            ty = y + a + (-10 if a < 0 else 20)
            parts.append(
                f'<text x="{cx}" y="{ty}" text-anchor="middle" '
                f'font-size="19" fill="currentColor">{w}</text>'
            )
    for i in range(len(slots) + 1):
        x = x0 + i * seg_w
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" '
            f'stroke="currentColor" stroke-width="2"/>'
        )
        if i == len(slots):
            parts.append(
                f'<circle cx="{x}" cy="{y}" r="{r - 4}" fill="none" '
                f'stroke="currentColor" stroke-width="2"/>'
            )
        parts.append(
            f'<text x="{x}" y="{y + 5}" text-anchor="middle" font-size="14" '
            f'fill="currentColor">{i}</text>'
        )
    return (
        '<svg viewBox="0 0 1100 300" role="img" '
        'aria-label="GRID grammar as a linear FST" '
        'style="width:100%;height:auto" xmlns="http://www.w3.org/2000/svg">'
        '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#888"/></marker></defs>'
        + "".join(parts) + "</svg>"
    )


def roi_strip(roi: np.ndarray, max_frames: int = 16) -> np.ndarray:
    """Sample up to max_frames evenly and tile them into one (H, W*k) image."""
    T = roi.shape[0]
    idx = np.linspace(0, T - 1, min(T, max_frames)).round().astype(int)
    return np.concatenate([roi[i] for i in idx], axis=1)


def _video_fps(video: str) -> float:
    import cv2

    cap = cv2.VideoCapture(video)
    try:
        return cap.get(cv2.CAP_PROP_FPS) or 0.0
    finally:
        cap.release()


def _video_duration(video: str) -> float:
    import cv2

    cap = cv2.VideoCapture(video)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps > 0 and 0 < frames < 1e7:
            return frames / fps
    finally:
        cap.release()
    return _ffprobe_duration(video)


def _ffprobe_duration(video: str) -> float:
    if shutil.which("ffprobe") is None:
        return 0.0
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "packet=pts_time", "-of", "csv=p=0", video],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0
    times = [float(t) for t in out.split() if t and t != "N/A"]
    return max(times) if times else 0.0


def _duration_error(video: str, max_duration: float = MAX_DURATION_S):
    duration = _video_duration(video)
    if not duration:
        return "Could not read this clip's length. Please try another clip (MP4 or WebM)."
    if not (
        MIN_DURATION_S - DURATION_TOL_S <= duration <= max_duration + DURATION_TOL_S
    ):
        return (
            f"Clip is {duration:.1f}s long; please use a clip between "
            f"{MIN_DURATION_S:g}s and {max_duration:g}s."
        )
    return None


def _resample_to_target_fps(src: str, max_seconds: float = None):
    """Re-encode at 25 fps. max_seconds truncates the output whatever the
    container header claims, which bounds the frames the tracker reads."""
    if shutil.which("ffmpeg") is None:
        return None
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    cap = ["-t", f"{max_seconds:g}"] if max_seconds else []
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, *cap, "-vf", f"fps={TARGET_FPS:g}",
             "-an", "-c:v", "mpeg4", "-q:v", "2", out],
            check=True, capture_output=True, timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        Path(out).unlink(missing_ok=True)
        return None
    return out


def _to_playable_mp4(src: str, out_dir=None):
    """Re-encode to H.264/yuv420p mp4 so browsers can play it (GRID's .mpg
    files play in OpenCV but not HTML5 <video>)."""
    if shutil.which("ffmpeg") is None:
        return None
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=out_dir).name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-movflags", "+faststart", out],
            check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, OSError):
        Path(out).unlink(missing_ok=True)
        return None
    return out


def _frames_to_mp4(frames: np.ndarray, fps: int = 25, scale: int = 3, out_dir=None):
    """Encode grayscale ROI frames (T,H,W uint8) to an H.264 mp4."""
    if shutil.which("ffmpeg") is None:
        return None
    T, H, W = frames.shape
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=out_dir).name
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
        "-s", f"{W}x{H}", "-r", str(fps), "-i", "pipe:0",
        "-vf", f"scale=iw*{scale}:ih*{scale}:flags=neighbor",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out,
    ]
    try:
        p = subprocess.run(
            cmd, input=np.ascontiguousarray(frames, dtype=np.uint8).tobytes(),
            capture_output=True,
        )
    except OSError:
        Path(out).unlink(missing_ok=True)
        return None
    if p.returncode != 0:
        Path(out).unlink(missing_ok=True)
        return None
    return out


def _prepare_clip(video: str, max_duration: float, source: str = "upload"):
    """Resample to 25fps if needed. Returns the path to process (caller
    deletes it if different from `video`), or raises ValueError.

    Uploads and webcam clips are always re-encoded with a length cap, since
    their headers are untrusted; bundled examples are used as they are."""
    err = _duration_error(video, max_duration)
    if err:
        raise ValueError(err)
    if source != "example":
        capped = _resample_to_target_fps(video, max_seconds=max_duration + DURATION_TOL_S)
        if capped is None:
            raise ValueError("Could not read this clip. Please try another clip (MP4 or WebM).")
        return capped
    fps = _video_fps(video)
    if fps and abs(fps - TARGET_FPS) > 0.1:
        resampled = _resample_to_target_fps(video)
        if resampled is not None:
            logger.info(f"Resampled {video}: {fps:.2f} -> {TARGET_FPS:g} fps")
            return resampled
        logger.warning(f"{video} is {fps:.2f} fps and could not be resampled; "
                        "recognition accuracy may degrade.")
    return video


# Per-request temp dirs for UI output files; bounds disk use on a long-running
# server the same way icefall's demo.py does.
_REQUEST_DIRS: list = []
_MAX_REQUEST_DIRS = 4


def _new_request_dir() -> str:
    while len(_REQUEST_DIRS) >= _MAX_REQUEST_DIRS:
        shutil.rmtree(_REQUEST_DIRS.pop(0), ignore_errors=True)
    d = tempfile.mkdtemp(prefix="vsr-demo-")
    _REQUEST_DIRS.append(d)
    return d


def launch_ui(reader: KaldiLipReader, example_paths=None):
    import gradio as gr

    def _highlighted(word_confs):
        return [
            (f"{w.title()} {c:.2f}",
             "high" if c >= 0.9 else "medium" if c >= 0.7 else "low")
            for w, c in word_confs
        ] or None

    def infer(video, source, noise, blur, use_icefall=False):
        """Generator, not a plain function: yields the transcript as soon
        as decode finishes (raw MBR confidence -- fast but scores most
        correct words a flat, uninformative 1.00), then yields again once
        confidence calibration finishes (~1-2s more, see
        KaldiLipReader.refine_confidence) with just the confidence display
        updated in place. The text, ROI video/strip and buttons never wait
        on calibration -- only conf_out's second yield does."""
        if not video:
            yield "", None, None, None, gr.update(interactive=False), gr.update(), ""
            return
        # Count usage by clip source only (example / upload / webcam); no
        # IP, file name, or transcript, matching the privacy note in the UI.
        # Printed like [perf]: logger.info is silent after model load.
        backend = "icefall" if use_icefall else "kaldi"
        print(f"[usage] source={source} backend={backend}", file=sys.stderr, flush=True)
        try:
            max_dur = MAX_DURATION_WEBCAM_S if source == "webcam" else MAX_DURATION_S
            clip = _prepare_clip(video, max_dur, source)
            ood = ""
            try:
                if backend == "icefall":
                    text, roi, word_confs, ood = _recognize_icefall(clip, noise, blur)
                else:
                    text, roi, word_confs = reader.recognize(clip, noise, blur)
            finally:
                if clip != video:
                    Path(clip).unlink(missing_ok=True)
        except ValueError as e:
            yield "", None, None, None, gr.update(), None, str(e)
            return
        except RuntimeError:
            if backend != "icefall":
                logger.exception(f"Recognition failed for {video}")
                yield ("", None, None, None, gr.update(), None,
                       "Recognition failed. Please try a different clip -- a "
                       "frontal talking face, whole face visible, 2-15s long.")
                return
            logger.exception("icefall back-end failed")
            yield ("", None, None, None, gr.update(), None,
                   "The icefall model is unavailable right now. Please try "
                   "again shortly, or turn off the icefall toggle to use Kaldi.")
            return
        except Exception:
            logger.exception(f"Recognition failed for {video}")
            yield ("", None, None, None, gr.update(), None,
                   "Recognition failed. Please try a different clip -- a "
                   "frontal talking face, whole face visible, 2-15s long.")
            return
        if ood == "hide":
            # icefall's out-of-domain detector: definitely not a GRID
            # sentence, so show nothing rather than a confident guess.
            yield ("", None, None, None, gr.update(interactive=False), gr.update(),
                   "This doesn't look like a GRID sentence, so there's "
                   "nothing to show. Please speak a sentence in the fixed "
                   "command·colour·preposition·letter·digit·adverb pattern.")
            return
        out_dir = _new_request_dir()
        yield (
            text.title(),
            _highlighted(word_confs),
            _frames_to_mp4(roi, out_dir=out_dir),
            roi_strip(roi),
            gr.update(interactive=False),
            gr.update(),
            "",
        )
        # icefall's confidences are already calibrated by the service (TruCLeS
        # CEM fusion, isotonic fallback); only the Kaldi path refines them again.
        calibrated = reader.refine_confidence() if backend == "kaldi" else None
        if calibrated is not None:
            yield (gr.update(), _highlighted(calibrated), gr.update(),
                   gr.update(), gr.update(), gr.update(), gr.update())

    examples = []
    for pth in example_paths or []:
        if not Path(pth).exists():
            logger.warning(f"Example clip not found, skipping: {pth}")
            continue
        examples.append([_to_playable_mp4(str(pth)) or str(pth)])

    css = """
    .source-selection { height: var(--size-12) !important; flex-shrink: 0; }
    .source-selection .icon { width: 30px !important; height: 30px !important; }
    video { object-fit: contain !important; }
    .word-conf span { display: inline-block; white-space: nowrap; }
    .rec-text textarea { font-weight: 700; font-size: 1.5em; }
    .gradio-container { --block-title-text-weight: 700;
        --block-label-text-weight: 700; }
    .privacy-note p { color: var(--color-accent); font-weight: 600;
        margin: 4px 0; }
    #video-wrap { position: relative; }
    .status-msg { position: absolute; left: 0; right: 0; top: 36px;
        z-index: 20; pointer-events: none; }
    .status-msg p { margin: 0; padding: 10px 14px;
        background: rgba(0, 0, 0, 0.72); color: #fff; font-weight: 600;
        text-align: center; border-radius: var(--radius-lg) var(--radius-lg) 0 0; }
    footer { display: none !important; }
    /* Recognizer toggle: render the checkbox as a switch. */
    #backend-toggle input[type=checkbox] {
        appearance: none; -webkit-appearance: none; position: relative;
        width: 2.6em !important; height: 1.4em !important;
        border-radius: 0.7em !important; cursor: pointer;
        background-color: #9ca3af !important; background-image: none !important;
        border: none !important; transition: background 0.2s; flex-shrink: 0; }
    #backend-toggle input[type=checkbox]::before {
        content: ""; position: absolute; top: 0.15em; left: 0.15em;
        width: 1.1em; height: 1.1em; border-radius: 50%; background: #fff;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.3); transition: transform 0.2s; }
    #backend-toggle input[type=checkbox]:checked {
        background-color: var(--color-accent) !important; background-image: none !important; }
    #backend-toggle input[type=checkbox]:checked::before { transform: translateX(1.2em); }
    button[aria-label="Trim video to selection"] { display: none !important; }
    .wrap .error { display: none !important; }
    """
    # Grammar-hint ticker over the empty upload area (rotates a random
    # grammar-valid example sentence) and an auto-reset for components stuck
    # in a client-side error state. Copied from icefall's demo.py.
    js = """
    () => {
      const cmd = ['bin','lay','place','set'],
            col = ['blue','green','red','white'],
            prep = ['at','by','in','with'],
            letters = 'abcdefghijklmnopqrstuvxyz'.split(''),
            dig = ['zero','one','two','three','four','five','six','seven',
                   'eight','nine'],
            adv = ['again','now','please','soon'];
      const pick = a => a[Math.floor(Math.random() * a.length)];
      const cap = w => w.charAt(0).toUpperCase() + w.slice(1);
      const ticker = document.getElementById('sentence-ticker');
      if (ticker && !ticker.dataset.init) {
        ticker.dataset.init = '1';
        ticker.innerHTML =
          'A valid sentence uses a 6-word grammatical structure: '
          + 'command | colour | preposition | letter | number | adverb, '
          + 'such as \\u201C<span id="ticker-eg"></span>\\u201D';
        const eg = document.getElementById('ticker-eg');
        eg.style.transition = 'opacity 0.35s';
        const tick = () => {
          eg.style.opacity = 0;
          setTimeout(() => {
            eg.textContent = [pick(cmd), pick(col), pick(prep),
              pick(letters), pick(dig), pick(adv)].map(cap).join(' ');
            eg.style.opacity = 1;
          }, 350);
        };
        tick();
        setInterval(tick, 3500);
      }
      const toggleHint = () => {
        const vin = document.getElementById('video-input');
        if (vin && ticker) {
          ticker.style.display = vin.querySelector('video') ? 'none' : '';
        }
      };
      const resetErrored = () => {
        document.querySelectorAll('.block .error').forEach(err => {
          const block = err.closest('.block');
          if (!block) return;
          const btn = block.querySelector('.clear-status button')
                   || block.querySelector('button.clear-status');
          if (btn) btn.click();
        });
      };
      const onMutate = () => { resetErrored(); toggleHint(); };
      onMutate();
      new MutationObserver(onMutate)
        .observe(document.body, {subtree: true, childList: true});
    }
    """
    theme = gr.themes.Default(primary_hue="indigo")
    with gr.Blocks(
        title="Can AI Read Your Lips? A Live Lipreading Demo "
              "(based on GRID corpus, Kaldi and icefall backends)",
        delete_cache=(600, 900), css=css, js=js, theme=theme,
    ) as demo:
        gr.Markdown("# Can AI Read Your Lips?")
        gr.Markdown(
            "**Live visual speech recognition constrained to the GRID "
            "corpus grammar.**",
            elem_classes=["subtitle"],
        )
        with gr.Row():
            with gr.Column():
                with gr.Group(elem_id="video-wrap"):
                    video_in = gr.Video(
                        label="Frontal talking-face clip",
                        height="min(410px, 45vh)",
                        elem_id="video-input",
                        sources=["webcam", "upload"],
                    )
                    status_msg = gr.Markdown("", elem_classes=["status-msg"])
                backend_rd = gr.Checkbox(
                    value=False, label="Use the icefall model",
                    info="Off: Kaldi TDNN-F chain (default). "
                         "On: icefall Conformer-CTC.",
                    elem_id="backend-toggle",
                )
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary", interactive=False)
                    new_clip = gr.Button("Try another clip", interactive=False)
                gr.HTML(
                    '<div id="sentence-ticker" style="text-align:center;'
                    'font-weight:600;padding:6px 4px;min-height:1.4em;'
                    'color:var(--color-accent)"></div>'
                )
                gr.Markdown(
                    "🔒 *Your clip and the mouth crops shown here stay in "
                    "temporary memory-backed storage and are deleted "
                    "automatically within 30 minutes. Transcripts are never "
                    "logged.*",
                    elem_classes=["privacy-note"],
                )
                gr.Markdown(
                    "*Best with a single frontal face in good lighting "
                    f"({MIN_DURATION_S:g}–{MAX_DURATION_WEBCAM_S:g} s).*"
                )
                source_state = gr.State("upload")
                if examples:
                    example_set = gr.Examples(
                        examples=examples, inputs=[video_in],
                        label="Example clips from speakers unseen in "
                              "training: s1, s2, s20, s22 & one from "
                              "Lombard Grid corpus & a live recording "
                              "(click & submit)",
                    )
                    # Clicking an example fires neither upload nor
                    # stop_recording, so tag the source here.
                    example_set.load_input_event.then(
                        lambda: "example", outputs=[source_state])
                with gr.Accordion(
                    "Challenge the model: add noise or blur (optional)", open=False,
                ):
                    gr.Markdown(
                        "These sliders worsen the video quality of the mouth "
                        "region before the model reads it. The Mouth ROI panels "
                        "show exactly what the model received. Leave at 0 for "
                        "normal recognition."
                    )
                    noise_sl = gr.Slider(0, 50, value=0, step=1, label="Noise (0 = none)")
                    blur_sl = gr.Slider(0, 5, value=0, step=0.25, label="Blur (0 = none)")
            with gr.Column(elem_id="results-col"):
                with gr.Row():
                    roi_vid = gr.Video(label="Mouth ROI (animated)",
                                       show_download_button=False, height=240, scale=1)
                    with gr.Column(scale=2):
                        text_out = gr.Textbox(label="Prediction", elem_classes=["rec-text"])
                        conf_out = gr.HighlightedText(
                            label="Word confidence",
                            color_map={"high": "green", "medium": "yellow", "low": "red"},
                            show_inline_category=False, show_legend=False,
                            elem_classes=["word-conf"],
                        )
                strip_out = gr.Image(label="Mouth ROI (sampled frames)",
                                     image_mode="L", height=140,
                                     show_download_button=False)
                gr.Markdown(
                    "### Valid GRID sentence structure\n"
                    "The model only understands 6-word sentences that follow the "
                    "pattern below, one word from each column, e.g. "
                    "**“place green with h eight now”** or "
                    "**“bin blue at f two now”**."
                )
                gr.HTML(_grammar_fst_svg())
                with gr.Accordion("Full grammar description", open=False):
                    gr.Markdown(GRID_GRAMMAR_MD)
        gr.Markdown(ACKNOWLEDGEMENTS_MD)

        def check_length(video, source):
            """Reject out-of-range clips as soon as they are supplied, before
            Submit: clear the input and explain why inline. recognize() and
            infer() re-check as a backstop."""
            if video:
                max_dur = MAX_DURATION_WEBCAM_S if source == "webcam" else MAX_DURATION_S
                try:
                    err = _duration_error(video, max_dur)
                except Exception:
                    logger.exception(f"Could not read clip {video}")
                    return (None, source, "This video could not be read; "
                            "please try a different file or recording.")
                if err:
                    return None, source, err
            return gr.update(), source, ""

        video_in.upload(
            lambda v: check_length(v, "upload"),
            inputs=[video_in], outputs=[video_in, source_state, status_msg],
        )
        video_in.stop_recording(
            lambda v: check_length(v, "webcam"),
            inputs=[video_in], outputs=[video_in, source_state, status_msg],
        )

        def on_clip_change(video):
            loaded = video is not None
            return (gr.update(interactive=loaded), gr.update(interactive=loaded),
                    "", None, None, None)

        video_in.change(
            on_clip_change, inputs=[video_in],
            outputs=[submit, new_clip, text_out, conf_out, roi_vid, strip_out],
        )
        new_clip.click(lambda: (None, ""), outputs=[video_in, status_msg])
        for slider in (noise_sl, blur_sl, backend_rd):
            slider.change(lambda v: gr.update(interactive=v is not None),
                          inputs=[video_in], outputs=[submit])

        submit.click(
            infer, inputs=[video_in, source_state, noise_sl, blur_sl, backend_rd],
            outputs=[text_out, conf_out, roi_vid, strip_out, submit, video_in, status_msg],
        ).then(
            None, None, None,
            js="""() => {
              setTimeout(() => {
                const col = document.getElementById('results-col');
                const txt = document.querySelector('.rec-text textarea, .rec-text input');
                if (!col || !txt || !txt.value.trim()) return;
                const top = col.getBoundingClientRect().top;
                if (top >= 0 && top < window.innerHeight * 0.5) return;
                col.scrollIntoView({behavior: 'smooth', block: 'start'});
              }, 150);
            }""",
        )

    demo.queue(default_concurrency_limit=1, max_size=10)
    demo.launch(server_name="0.0.0.0", max_file_size="100mb", show_api=False)


def _find_example_clips():
    """4 unseen-speaker GRID test clips + one Lombard GRID clip + one live
    webcam recording, mirroring icefall's demo default --examples list.
    Missing files are skipped (launch_ui already does this)."""
    return [
        str(RECIPE_ROOT / "grid-corpus/s1/bbaf2n.mpg"),
        str(RECIPE_ROOT / "grid-corpus/s2/sgbp4s.mpg"),
        str(RECIPE_ROOT / "grid-corpus/s20/pgwj3p.mpg"),
        str(RECIPE_ROOT / "grid-corpus/s22/srwaza.mpg"),
        str(RECIPE_ROOT / "demo/s3_l_lgin3a.mov"),
        str(RECIPE_ROOT / "demo/recorded_demo_lgwf8a.mp4"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?", help="video path (CLI mode)")
    parser.add_argument("--ui", action="store_true", help="launch the Gradio web UI")
    parser.add_argument("--fp16", action="store_true",
                        help="run AV-HuBERT in half precision (experimental).")
    parser.add_argument("--idle-offload-seconds", type=int, default=120,
                        help="Move AV-HuBERT to CPU after this many idle "
                             "seconds, restoring it on the next request "
                             "(<1s). 0 disables. Default: %(default)s")
    args = parser.parse_args()

    reader = KaldiLipReader(fp16=args.fp16,
                            idle_offload_seconds=args.idle_offload_seconds)
    if args.ui:
        launch_ui(reader, example_paths=_find_example_clips())
    else:
        if not args.video:
            parser.error("video is required unless --ui is given")
        text, _, word_confs = reader.recognize(args.video)
        print(text)
        print(" ".join(f"{w}({c:.2f})" for w, c in word_confs))


if __name__ == "__main__":
    main()
