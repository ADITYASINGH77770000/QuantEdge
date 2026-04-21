"""Prediction backend for multi-model stock forecasting."""


def run_multi_model_prediction(*args, **kwargs):
    """Lazy import to keep utility modules lightweight during testing."""
    from .pipeline import run_multi_model_prediction as _run_multi_model_prediction

    return _run_multi_model_prediction(*args, **kwargs)


def rerun_prediction_inference(*args, **kwargs):
    """Reuse trained prediction bundles against refreshed market data."""
    from .pipeline import rerun_prediction_inference as _rerun_prediction_inference

    return _rerun_prediction_inference(*args, **kwargs)


__all__ = ["run_multi_model_prediction", "rerun_prediction_inference"]
