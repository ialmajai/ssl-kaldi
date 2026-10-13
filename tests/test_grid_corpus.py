"""Tests for egs/grid/s5/local/download_grid_corpus.py.

Parsed with ast rather than imported: the module needs tqdm and network
tooling that CI does not install, but the speaker remapping is pure data
and its failure mode (a missing speaker) is silent data loss.
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


def test_speaker_fix_map_is_a_complete_permutation():
    """The map rewrites the alignments tree speaker-by-speaker, then the
    original tree is deleted. A speaker missing from the keys is never
    copied and is lost; a speaker missing from the values is never a
    source. It must permute all 34 GRID speakers."""
    m = _speaker_fix_map()
    expected = {f"s{i}" for i in range(1, 35)}
    assert set(m) == expected
    assert set(m.values()) == expected
