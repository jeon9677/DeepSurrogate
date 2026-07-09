from .models import get_model_deepsurrogate
from .train import train_model, mc_predict

__all__ = [
    "get_model_deepsurrogate",
    "train_model",
    "mc_predict",
]

__version__ = "0.1.0"
