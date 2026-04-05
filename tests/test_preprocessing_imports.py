"""Regression tests for vlm_engine.preprocessing (e.g. torchvision API compatibility)."""


def test_preprocessing_module_imports() -> None:
    """Torchvision 0.26+ removed VideoReader from torchvision.io; preprocessing must still import."""
    from vlm_engine.preprocessing import (
        crop_black_bars_lr,
        get_video_metadata,
        preprocess_video,
    )

    assert callable(crop_black_bars_lr)
    assert callable(get_video_metadata)
    assert callable(preprocess_video)
