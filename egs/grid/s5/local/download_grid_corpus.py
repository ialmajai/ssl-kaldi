#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Based on code from Lhotse grid recipe
# https://github.com/lhotse-speech/lhotse/blob/master/lhotse/recipes/grid.py         


"""
The Grid Corpus is a large multitalker audiovisual sentence corpus designed to support joint
computational-behavioral studies in speech perception. In brief, the corpus consists of high-quality
audio and video (facial) recordings of 1000 sentences spoken by each of 34 talkers (18 male, 16 female),
for a total of 34000 sentences. Sentences are of the form "put red at G9 now".

Source: https://zenodo.org/record/3625687
"""
import os
import shutil
import subprocess
import tempfile
import zipfile
from importlib.util import find_spec
from pathlib import Path
from tqdm.auto import tqdm
from typing import Union
import argparse

Pathlike = Union[Path, str]

GRID_ZENODO_ID = "10.5281/zenodo.3625687"

def is_module_available(module_name: str) -> bool:
    """Check if a module is available."""
    return find_spec(module_name) is not None

def download_grid(
    target_dir: Pathlike = ".",
    force_download: bool = False,
) -> Path:
    """
    Download and untar the dataset
    """
    if not is_module_available("zenodo_get"):
        raise RuntimeError(
            "To download Grid Audio-Visual Speech Corpus please 'pip install zenodo_get'."
        )

    corpus_dir = Path(target_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    download_marker = corpus_dir / ".downloaded"
    if not download_marker.exists() or force_download:
        subprocess.run(
            f"zenodo_get {GRID_ZENODO_ID}", shell=True, check=True, cwd=corpus_dir
        )
        download_marker.touch()

    for p in tqdm(corpus_dir.glob("*.zip"), desc="Unzipping files"):
        with zipfile.ZipFile(p) as f:
            f.extractall(corpus_dir)

    # Speaker mapping to fix mis-assigned alignment data
    speaker_fix_map = {
        "s1": "s1",
        "s2": "s2",
        "s3": "s3",
        "s4": "s4",
        "s5": "s6",
        "s6": "s5",
        "s7": "s7",
        "s8": "s8",
        "s9": "s9",
        "s10": "s13",
        "s11": "s10",
        "s12": "s11",
        "s13": "s12",
        "s14": "s15",
        "s15": "s14",
        "s16": "s16",
        "s17": "s17",
        "s18": "s19",
        "s19": "s18",
        "s20": "s21",
        "s22": "s23",
        "s23": "s22",
        "s24": "s24",
        "s25": "s25",
        "s26": "s27",
        "s27": "s26",
        "s28": "s29",
        "s29": "s28",
        "s30": "s30",
        "s31": "s31",
        "s32": "s33",
        "s33": "s32",
        "s34": "s34",
    }

    # Downloaded alignment folder has mis-assigned speaker folders, we fix it here
    input_dir = corpus_dir / "alignments"
    tempfile.tempdir = os.path.abspath(corpus_dir)
    temp_alignment_dir = tempfile.mkdtemp()

    for tgt_folder, src_folder in speaker_fix_map.items():
        src_path = os.path.join(input_dir, src_folder)
        tgt_path = os.path.join(temp_alignment_dir, tgt_folder)
        shutil.copytree(src_path, tgt_path)
        print(f"Copied entire folder from {src_folder} to {tgt_folder}")

    shutil.rmtree(input_dir)
    os.rename(temp_alignment_dir, input_dir)
    return corpus_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the Grid Audio-Visual Speech Corpus"
    )
    parser.add_argument("--dir", required=True)
    download_grid(parser.parse_args().dir)