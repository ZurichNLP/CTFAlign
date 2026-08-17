from .base import EncodedText, Encoder

__all__ = ["EncodedText", "Encoder"]


def __getattr__(name):
    # Lazy import so transformers is only loaded when an HF encoder is used.
    if name == "HFEncoder":
        from .huggingface import HFEncoder
        return HFEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
