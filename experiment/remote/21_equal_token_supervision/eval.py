"""Reuse experiment 17's vLLM evaluation for experiment 21 adapters."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


LEGACY_EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "17_llm_clean"
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LEGACY_EXPERIMENT_DIR))

spec = importlib.util.spec_from_file_location(
    "_experiment_17_eval",
    LEGACY_EXPERIMENT_DIR / "eval.py",
)
if spec is None or spec.loader is None:
    raise ImportError("could not load experiment 17 evaluator")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

perform_eval = module.perform_eval
