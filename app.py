"""Urdu <-> English Neural Machine Translation - Streamlit app."""
import os

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(page_title="Urdu ↔ English Translator", page_icon="🌐", layout="centered")

# ---------------------------------------------------------------------------
# Configuration
# Set HF_USERNAME in Streamlit Cloud > App settings > Secrets:
#   HF_USERNAME = "your-huggingface-username"
# If your fine-tuned repos are not found, the app falls back to the
# pretrained Helsinki-NLP base models so it never crashes.
# ---------------------------------------------------------------------------
def get_username() -> str:
    try:
        if "HF_USERNAME" in st.secrets:
            return st.secrets["HF_USERNAME"]
    except Exception:
        pass
    return os.environ.get("HF_USERNAME", "YOUR_HUGGINGFACE_USERNAME")


HF_USERNAME = get_username()

MODELS = {
    "Urdu → English": {
        "finetuned": f"{HF_USERNAME}/urdu-english-translation",
        "base": "Helsinki-NLP/opus-mt-ur-en",
    },
    "English → Urdu": {
        "finetuned": f"{HF_USERNAME}/english-urdu-translation",
        "base": "Helsinki-NLP/opus-mt-en-ur",
    },
}

EXAMPLES = {
    "Urdu → English": ["پاکستان ایک خوبصورت ملک ہے۔", "آپ کا نام کیا ہے؟", "میں کل لاہور جا رہا ہوں۔"],
    "English → Urdu": ["Pakistan is a beautiful country.", "What is your name?", "I am going to Lahore tomorrow."],
}


@st.cache_resource(show_spinner=False)
def load_model(direction: str):
    """Load the fine-tuned model; fall back to the pretrained base model."""
    cfg = MODELS[direction]
    for repo, label in ((cfg["finetuned"], "fine-tuned"), (cfg["base"], "base")):
        try:
            tok = AutoTokenizer.from_pretrained(repo)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(repo)
            mdl.eval()
            return tok, mdl, repo, label
        except Exception:
            continue
    raise RuntimeError("Could not load any model.")


def translate(text, tok, mdl, num_beams=4):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_length=128, num_beams=num_beams)
    return tok.decode(out[0], skip_special_tokens=True)


# ------------------------------- UI ----------------------------------------
st.markdown(
    """
    <style>
    textarea[aria-label="Result"] {font-size: 1.1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌐 Urdu ↔ English Translator")
st.caption("Fine-tuned Hugging Face Marian models • trained on the OPUS-100 en-ur parallel corpus")

with st.sidebar:
    st.header("Settings")
    num_beams = st.slider("Beam size (quality vs speed)", 1, 8, 4)
    st.markdown("**Dataset:** [Helsinki-NLP/opus-100 (en-ur)](https://huggingface.co/datasets/Helsinki-NLP/opus-100)")
    st.markdown("**Base models:** [ur-en](https://huggingface.co/Helsinki-NLP/opus-mt-ur-en), [en-ur](https://huggingface.co/Helsinki-NLP/opus-mt-en-ur)")

direction = st.radio("Translation direction", list(MODELS.keys()), horizontal=True)

example = st.selectbox("Try an example", [""] + EXAMPLES[direction])
text = st.text_area("Enter text", value=example, height=150, placeholder="Type your sentence here...", max_chars=1000)

if st.button("Translate", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Loading model and translating (first run may take a minute)..."):
            tok, mdl, repo, label = load_model(direction)
            result = translate(text.strip(), tok, mdl, num_beams)
        st.success("Translation")
        if direction == "English → Urdu":
            st.markdown(f"<div dir='rtl' style='font-size:1.4rem'>{result}</div>", unsafe_allow_html=True)
        else:
            st.write(result)
        st.code(result, language=None)
        st.caption(f"Model used: `{repo}` ({label})")
