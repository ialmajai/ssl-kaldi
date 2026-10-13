"""Reading audio lists and writing Kaldi archives, with no Kaldi install.

kaldiio writes ark/scp in pure Python, so the output of this package is
consumable by Kaldi, ESPnet or anything else that reads the format, on a
machine that has never compiled Kaldi.
"""
import logging
import os
import subprocess
import wave

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _load_wav_scp_entry(rest):
    if rest.endswith("|"):
        raw = subprocess.run(
            rest[:-1], shell=True, capture_output=True, check=True
        ).stdout
        return _decode_wav_bytes(raw)
    return read_audio(rest)


def read_wav_scp(path):
    """Yield ``(utt_id, load)`` from a Kaldi wav.scp; ``load()`` returns
    ``(waveform, sample_rate)``.

    Handles both plain paths and the extended pipe form
    ``utt_id sox foo.flac -t wav - |``, which Kaldi recipes use heavily.
    Reading is deferred to ``load()`` so the caller can count an unreadable
    file or failing pipe as one failure instead of ending the job.
    """
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, rest = line.split(None, 1)
            yield utt_id, (lambda rest=rest: _load_wav_scp_entry(rest))


def read_file_list(path):
    """Yield ``(utt_id, load)`` from a list of audio paths; ``load()`` returns
    ``(waveform, sample_rate)``.

    The utterance id is the basename without extension, which is what most
    people expect when they point this at a directory listing.
    """
    with open(path) as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            yield (os.path.splitext(os.path.basename(p))[0],
                   (lambda p=p: read_audio(p)))


def read_audio(path):
    """Read a wav file. Falls back to soundfile for anything else."""
    if path.lower().endswith(".wav"):
        try:
            with wave.open(path, "rb") as w:
                if w.getsampwidth() != 2:
                    raise wave.Error("not 16-bit")
                sr = w.getframerate()
                data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                if w.getnchannels() > 1:
                    data = data.reshape(-1, w.getnchannels())
                return data, sr
        except wave.Error:
            pass
    try:
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError(
            f"cannot read {path}: install soundfile for non-PCM16 or non-wav audio"
        ) from e
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    return data, sr


def _decode_wav_bytes(raw):
    """Decode piped wav bytes. Only 16-bit PCM is read as int16; anything else
    (24/32-bit, float) goes through soundfile, as read_audio does for files."""
    import io as _io

    with wave.open(_io.BytesIO(raw), "rb") as w:
        if w.getsampwidth() == 2:
            sr = w.getframerate()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            if w.getnchannels() > 1:
                data = data.reshape(-1, w.getnchannels())
            return data, sr
    try:
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError("piped audio is not 16-bit PCM: install soundfile to read it") from e
    data, sr = sf.read(_io.BytesIO(raw), dtype="float32", always_2d=False)
    return data, sr


def write_side_files(out_dir, utt2num_frames, utt2dur, frame_shift):
    """Write the metadata a Kaldi data directory expects alongside feats.scp."""
    with open(os.path.join(out_dir, "utt2num_frames"), "w") as f:
        for utt, n in sorted(utt2num_frames.items()):
            f.write(f"{utt} {n}\n")
    with open(os.path.join(out_dir, "utt2dur"), "w") as f:
        for utt, d in sorted(utt2dur.items()):
            f.write(f"{utt} {d:.3f}\n")
    with open(os.path.join(out_dir, "frame_shift"), "w") as f:
        f.write(f"{frame_shift}\n")
