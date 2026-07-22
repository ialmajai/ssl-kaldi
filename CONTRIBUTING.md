# Contributing to ssl-kaldi

Thanks for your interest in improving **ssl-kaldi**. Contributions of all kinds
are welcome: new recipes, bug fixes, better documentation, or results.

## Reporting issues

Open a GitHub issue with:

- what you ran (recipe, `ssl_model`, layer, stage),
- what you expected vs. what happened,
- the relevant log excerpt (e.g. `exp/.../log/...` or `data/.../log/make_ssl_*.log`),
- your environment (OS, Python, `torch`/`transformers` versions, GPU).

## Development setup

```
conda create -n ssl-kaldi python=3.8 -y
conda activate ssl-kaldi
pip install -r requirements.txt        # or requirements.lock for a pinned, reproducible install
```

The GRID lipreading recipe additionally needs AV-HuBERT / fairseq / dlib
(`egs/grid/s5/requirements.txt`).

A working [Kaldi](https://github.com/kaldi-asr/kaldi) checkout is required; each
recipe expects the usual `steps`, `utils`, `shared`, and `conf` symlinks.

### Dependency lock

`requirements.txt` is the loose, human-edited spec; `requirements.lock` pins the
full dependency closure to tested versions for reproducible installs
(`pip install -r requirements.lock`). It is platform-specific
(linux/x86_64, CUDA 12). After editing `requirements.txt`, regenerate the lock
from a validated Python 3.8 environment:

```
pip install pip-tools
pip-compile requirements.txt -o requirements.lock
```

## Code style

- CI runs `flake8` for **syntax errors and undefined names** (`E9,F63,F7,F82`).
  Keep Python parse-clean; match the style of the surrounding code.
- Target **Python 3.8**, the floor for the whole repo (the GRID recipe's
  AV-HuBERT / fairseq stack requires it). If you use newer typing syntax
  (`list[...]`, `X | Y`), guard the module with
  `from __future__ import annotations`.
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
