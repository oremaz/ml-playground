"""Optional Weights & Biases logging helpers shared by BenchmarkRunner and notebooks.

All helpers degrade gracefully: without wandb installed or without credentials
they either no-op or fall back to offline mode, so pipelines never fail because
of logging.
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "bench-research-ml"


def wandb_available() -> bool:
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        return False


def init_wandb_run(
    project: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    group: Optional[str] = None,
):
    """Start a W&B run; returns None (and logs a warning) if wandb is unusable.

    Mode resolution: WANDB_MODE env var wins; otherwise online when an API key
    is available (env or netrc), offline when not.
    """
    try:
        import wandb
    except ImportError:
        logger.warning("wandb is not installed; skipping W&B logging")
        return None

    mode = os.environ.get("WANDB_MODE")
    if mode is None:
        has_key = bool(os.environ.get("WANDB_API_KEY")) or bool(getattr(wandb.api, "api_key", None))
        mode = "online" if has_key else "offline"

    try:
        return wandb.init(
            project=project or DEFAULT_PROJECT,
            name=name,
            config=config,
            tags=tags,
            group=group,
            mode=mode,
            reinit="create_new",
        )
    except TypeError:
        # Older wandb releases do not accept reinit="create_new"
        try:
            return wandb.init(project=project or DEFAULT_PROJECT, name=name, config=config,
                              tags=tags, group=group, mode=mode, reinit=True)
        except Exception as exc:
            logger.warning("wandb.init failed (%s); skipping W&B logging", exc)
            return None
    except Exception as exc:
        logger.warning("wandb.init failed (%s); skipping W&B logging", exc)
        return None


def _numeric_items(row: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for key, value in row.items():
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def log_history(run, history: Optional[List[Dict[str, Any]]]) -> None:
    """Log a pipeline training history (list of per-epoch or per-fold dicts)."""
    if run is None or not history:
        return
    try:
        for step, row in enumerate(history):
            metrics = _numeric_items(row)
            if metrics:
                run.log(metrics, step=step)
    except Exception as exc:
        logger.warning("wandb history logging failed: %s", exc)


def log_summary(run, summary: Optional[Dict[str, Any]]) -> None:
    if run is None or not summary:
        return
    try:
        for key, value in _numeric_items(summary).items():
            run.summary[key] = value
    except Exception as exc:
        logger.warning("wandb summary logging failed: %s", exc)


def finish_run(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        logger.warning("wandb finish failed: %s", exc)
