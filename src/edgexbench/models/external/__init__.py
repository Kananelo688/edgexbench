from . import shufflenetv1

def register_all() -> None:
    """Register every external architecture. Call once at startup."""
    shufflenetv1.register()