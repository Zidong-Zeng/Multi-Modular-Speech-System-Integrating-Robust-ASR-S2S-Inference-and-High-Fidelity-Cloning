# -*- coding: utf-8 -*-
"""Compatibility wrapper for the official C2 ASR entrypoint."""

from __future__ import annotations

import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ASSIGNMENT_C_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
C2_CODE_DIR = os.path.join(ASSIGNMENT_C_DIR, "C2_ASR", "code")
if C2_CODE_DIR not in sys.path:
    sys.path.insert(0, C2_CODE_DIR)

from c2_asr import *  # noqa: F401,F403
from c2_asr import main


if __name__ == "__main__":
    main()
