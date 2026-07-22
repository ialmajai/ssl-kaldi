# Contributing to ssl-kaldi

Thanks for your interest in improving **ssl-kaldi**. Contributions of all kinds
are welcome — new recipes, bug fixes, better documentation, or results.

## Reporting issues

Open a GitHub issue with:

- what you ran (recipe, `ssl_model`, layer, stage),
- what you expected vs. what happened,
- the relevant log excerpt (e.g. `exp/.../log/...` or `data/.../log/make_ssl_*.log`),
- your environment (OS, Python, `torch`/`transformers` versions, GPU).

## Development setup

```
conda create -n ssl-kaldi python=3.10 -y
conda activate ssl-kaldi
pip install -r requirements.txt
```

The GRID lipreading recipe is the exception: it needs a separate **Python 3.8**
environment (AV-HuBERT / fairseq / dlib) with `egs/grid/s5/requirements.txt`.

A working [Kaldi](https://github.com/kaldi-asr/kaldi) checkout is required; each
recipe expects the usual `steps`, `utils`, `shared`, and `conf` symlinks.

## Code style

- CI runs `flake8` for **syntax errors and undefined names** (`E9,F63,F7,F82`).
  Keep Python parse-clean; match the style of the surrounding code.
- Target **Python 3.10** for the audio recipes and shared code. Code under the
  GRID recipe must stay **Python 3.8**-compatible (its AV-HuBERT / fairseq stack);
  if you use newer typing syntax there (`list[...]`, `X | Y`), guard the module
  with `from __future__ import annotations`.
- Prefer reusing the shared building blocks in [`shared/`](shared/)
  (`make_ssl.sh`, `compute_ssl_feats.py`, `pca.py`, `interpolate.py`) over
  per-recipe copies.

## Adding a recipe

- Follow the standard Kaldi layout: `egs/<corpus>/s5/{local,conf,run*.sh}`.
- Extract SSL features via `shared/make_ssl.sh --ssl-model <hf-model> --layer N`
  rather than a recipe-local script.
- Add a `README.md` matching the other recipes: task description, setup, and a
  results table (with a conventional-features baseline for comparison).

## Pull requests

1. Branch off `master`.
2. Keep changes focused; every changed line should trace to the PR's purpose.
3. Make sure CI is green.
4. Describe what you changed and include before/after results where relevant.

By contributing, you agree that your contributions are licensed under the
[Apache 2.0](LICENSE) license.
