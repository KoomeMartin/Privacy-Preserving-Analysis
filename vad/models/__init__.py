"""models/__init__.py — unified model factory."""
from models.pel4vad.model import XModel, build_pel4vad
from models.mgfn.model    import MGFNModel, build_mgfn


def build_model(model_name: str, cfg: dict):
    """Return an initialised (un-trained) model on CPU."""
    name = model_name.lower()
    if name == "pel4vad":
        return build_pel4vad(cfg)
    elif name == "mgfn":
        return build_mgfn(cfg)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Use 'pel4vad' or 'mgfn'.")
