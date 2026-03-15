from configs.base import *
from configs.pel4vad_cfg import get_pel4vad_config
from configs.mgfn_cfg import get_mgfn_config

def get_config(model_name: str, overrides: dict = None) -> dict:
    """Return a config dict for the requested model."""
    model_name = model_name.lower()
    if model_name == "pel4vad":
        return get_pel4vad_config(overrides)
    elif model_name == "mgfn":
        return get_mgfn_config(overrides)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose 'pel4vad' or 'mgfn'.")
