import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def resolve_hf_token():
    """
    Returns the Hugging Face token from env/.env or None if not set.
    Also exports HUGGINGFACE_HUB_TOKEN so HF libs can auto-pick it up.
    """
    tok = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if tok:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = tok
    return tok
