"""Tests that need no model download, no GPU and no Kaldi."""
import numpy as np
import pytest

from ssl_kaldi_feats import preprocess_waveform
from ssl_kaldi_feats.cli import build_parser
from ssl_kaldi_feats.io import write_side_files


@pytest.mark.parametrize("shape", [(16000,), (16000, 1), (1, 16000)])
def test_mono_accepted(shape):
    assert preprocess_waveform(np.zeros(shape, np.int16)).dtype == np.float32


def test_stereo_rejected():
    """Left unchecked, np.squeeze passes (n, 2) through and the HF extractor
    reads it as a batch of two, returning wrong features silently."""
    with pytest.raises(ValueError, match="expected mono audio"):
        preprocess_waveform(np.zeros((16000, 2), np.int16))


def test_int_scaled_to_unit_range():
    peak = preprocess_waveform(np.array([np.iinfo(np.int16).max], np.int16))
    assert peak[0] == pytest.approx(1.0)


def test_float_passthrough():
    out = preprocess_waveform(np.array([0.5, -0.5], np.float64))
    assert out.dtype == np.float32 and out[0] == pytest.approx(0.5)


def test_layer_and_layers_are_exclusive():
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--model", "m", "--layer", "1", "--layers", "1,2",
                      "--wav-scp", "w", "--output-dir", "o"])


def test_a_source_is_required():
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--model", "m", "--layer", "1", "--output-dir", "o"])


def test_final_layer_norm_defaults_to_keep():
    p = build_parser()
    a = p.parse_args(["--model", "m", "--layer", "1", "--wav-scp", "w",
                      "--output-dir", "o"])
    assert a.final_layer_norm == "keep"


def test_side_files_are_sorted_and_formatted(tmp_path):
    write_side_files(str(tmp_path), {"b": 20, "a": 10}, {"b": 0.4, "a": 0.2}, 0.02)
    assert (tmp_path / "utt2num_frames").read_text() == "a 10\nb 20\n"
    assert (tmp_path / "utt2dur").read_text() == "a 0.200\nb 0.400\n"
    assert (tmp_path / "frame_shift").read_text() == "0.02\n"


def test_frame_shift_snaps_to_known_rates():
    """A measured 0.0201 must report as 0.02, not leak into frame_shift."""
    from ssl_kaldi_feats.extractor import SSLExtractor

    snap = SSLExtractor._measure_frame_shift
    known = SSLExtractor._KNOWN_FRAME_SHIFTS

    class Fake:
        _KNOWN_FRAME_SHIFTS = known
        _SNAP_TOLERANCE = SSLExtractor._SNAP_TOLERANCE
        model_id = "fake"
        layers = [1]

        def __init__(self, frames):
            self._frames = frames

        def extract(self, probe):
            import numpy as np
            return {1: np.zeros((self._frames, 4))}

    assert snap(Fake(199)) == 0.02      # one frame short of 200, still 20 ms
    assert snap(Fake(200)) == 0.02
    assert snap(Fake(100)) == 0.04      # a 40 ms encoder is reported as such
    assert snap(Fake(400)) == 0.01


def test_unknown_frame_rate_is_reported_not_snapped():
    """An 8 ms rate is outside tolerance of anything known: report it."""
    from ssl_kaldi_feats.extractor import SSLExtractor

    class Fake:
        _KNOWN_FRAME_SHIFTS = SSLExtractor._KNOWN_FRAME_SHIFTS
        _SNAP_TOLERANCE = SSLExtractor._SNAP_TOLERANCE
        model_id = "fake"
        layers = [1]

        def extract(self, probe):
            import numpy as np
            return {1: np.zeros((500, 4))}

    assert SSLExtractor._measure_frame_shift(Fake()) == pytest.approx(0.008)
