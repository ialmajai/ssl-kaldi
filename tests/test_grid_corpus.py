"""Tests for egs/grid/s5/local/download_grid_corpus.py.

Parsed with ast rather than imported: the module needs tqdm and network
tooling that CI does not install.
"""
import ast
import os

PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "egs", "grid", "s5", "local", "download_grid_corpus.py",
)


def _speaker_fix_map():
    tree = ast.parse(open(PATH).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            pairs = {}
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                    pairs[k.value] = v.value
            if pairs and all(str(k).startswith("s") for k in pairs):
                return pairs
    raise AssertionError("speaker_fix_map not found")


def test_speaker_fix_map_covers_every_video_speaker():
    """Check the map writes each of the 33 speakers with video exactly once.
    s21 has no video; its alignments (held in the s20 folder) are dropped."""
    m = _speaker_fix_map()
    speakers = {f"s{i}" for i in range(1, 35)}
    assert set(m) == speakers - {"s21"}
    assert set(m.values()) == speakers - {"s20"}
    assert len(set(m.values())) == len(m)
