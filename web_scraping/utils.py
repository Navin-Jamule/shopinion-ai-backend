def clean_text(text: str) -> str:
    """Basic text cleaning utility"""
    return text.strip().replace("\n", " ") if text else text