#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright   2025  (author: Ibrahim Almajai)
# Apache 2.0
"""Shared Kaldi I/O helpers for the SSL feature-extraction scripts."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def write_utt2dur(utt2dur: dict, out_spec: Optional[str]) -> None:
    """Write a utt2dur mapping to a Kaldi text file.

    ``out_spec`` may carry an ``ark,t:`` or ``ark:`` wspecifier prefix, which is
    stripped to obtain the destination path. A falsy spec is a no-op.
    """
    if not out_spec:
        return
    path = out_spec
    for prefix in ("ark,t:", "ark:"):
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    logger.info(f"Writing utt2dur: {path}")
    try:
        with open(path, "w") as f:
            for utt_id, dur in sorted(utt2dur.items()):
                f.write(f"{utt_id} {dur:.3f}\n")
        logger.info(f"Wrote {len(utt2dur)} durations")
    except OSError as e:
        logger.error(f"Failed to write utt2dur: {e}")
