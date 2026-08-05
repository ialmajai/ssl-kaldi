"""Tests for shared/, covering the failure modes that bit us in practice.

Deliberately free of Kaldi, GPU and network: CI has none of them. Every case
here is a bug that actually occurred and that `flake8 --select=E9,F63,F7,F82`
could not have caught.

Run with:  pytest -q tests/
"""
import importlib.util
import logging
import os
import sys

import numpy as np
import pytest

SHARED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "shared")
sys.path.insert(0, SHARED)


def _load(name):
    spec = importlib.util.spec_from_file_location(name,
                                                  os.path.join(SHARED, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kaldi_io_utils = _load("kaldi_io_utils")


# ---------------------------------------------------------------- waveforms

@pytest.fixture(scope="module")
def cf():
    # Imports torch and transformers, but downloads nothing.
    return _load("compute_ssl_feats")


@pytest.mark.parametrize("shape", [(16000,), (16000, 1), (1, 16000)])
def test_mono_waveforms_accepted(cf, shape):
    out = cf.preprocess_waveform(np.zeros(shape, dtype=np.int16))
    assert out.dtype == np.float32


def test_stereo_is_rejected_not_mangled(cf):
    """np.squeeze leaves (n, 2) alone and the HF extractor then reads it as a
    batch of two, returning wrong features with no error."""
    with pytest.raises(ValueError, match="expected mono audio"):
        cf.preprocess_waveform(np.zeros((16000, 2), dtype=np.int16))


def test_int_waveform_is_scaled_to_unit_range(cf):
    peak = cf.preprocess_waveform(np.array([np.iinfo(np.int16).max], dtype=np.int16))
    assert peak[0] == pytest.approx(1.0)


# ---------------------------------------------------------------- utt2dur

def test_write_utt2dur_strips_wspecifier(tmp_path):
    dest = tmp_path / "utt2dur"
    kaldi_io_utils.write_utt2dur({"b": 2.0, "a": 1.5}, f"ark,t:{dest}")
    assert dest.read_text() == "a 1.500\nb 2.000\n"


def test_write_utt2dur_is_sorted(tmp_path):
    dest = tmp_path / "utt2dur"
    kaldi_io_utils.write_utt2dur({"z": 1.0, "a": 1.0}, str(dest))
    assert [l.split()[0] for l in dest.read_text().splitlines()] == ["a", "z"]


def test_write_utt2dur_raises_instead_of_reporting_success(tmp_path):
    """Swallowing this let the caller log 'completed successfully' and exit 0,
    with the failure resurfacing later as a confusing missing-file error."""
    with pytest.raises(SystemExit):
        kaldi_io_utils.write_utt2dur({"a": 1.0},
                                     str(tmp_path / "no_such_dir" / "utt2dur"))


def test_write_utt2dur_noop_on_empty_spec(tmp_path):
    kaldi_io_utils.write_utt2dur({"a": 1.0}, None)  # must not raise


# ---------------------------------------------------------------- logging

def test_logger_survives_root_level_being_reset(cf):
    """transformers' from_pretrained() resets the ROOT logger to WARNING, which
    silently swallowed every INFO the extractor emitted after model load."""
    logging.getLogger().setLevel(logging.WARNING)
    assert cf.logger.isEnabledFor(logging.INFO)


# ---------------------------------------------------------------- pca

def test_flush_batch_skips_undersized_trailing_batch():
    pca = _load("pca")

    class FakeIPCA:
        n_components = 30
        device = "cpu"
        called = False

        def partial_fit(self, x):
            self.called = True

    ipca = FakeIPCA()
    pca._flush_batch([np.zeros((5, 40), dtype=np.float32)], ipca)
    assert not ipca.called, "should skip rather than fail inside torchdr"

    ipca2 = FakeIPCA()
    pca._flush_batch([np.zeros((100, 40), dtype=np.float32)], ipca2)
    assert ipca2.called


# ---------------------------------------------------------------- cli

def test_layers_requires_out_template(cf):
    sys.argv = ["x", "--layers", "1,2", "in", "out"]
    with pytest.raises(SystemExit, match="out-template"):
        cf.process_features(cf.parse_args())


def test_final_layer_norm_defaults_to_keep(cf):
    sys.argv = ["x", "in", "out"]
    assert cf.parse_args().final_layer_norm == "keep"
