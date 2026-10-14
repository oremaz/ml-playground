"""Compatibility import for the historical module path."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("ml_pipeline.pipelines_torch.ssl_wrappers")
