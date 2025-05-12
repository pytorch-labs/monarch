# pyre-unsafe
def is_te_available() -> bool:
    try:
        import transformer_engine  # noqa: F401

        return True
    except ImportError:
        return False
