# Urdu ↔ English Neural Machine Translation

End-to-end NLP project: fine-tune pretrained Hugging Face translation models on an online parallel corpus and deploy them with Streamlit.

## Problem statement
Many users need quick translation between Urdu and English. This project fine-tunes pretrained sequence-to-sequence models for both directions and serves them through a simple web app.

## Dataset (online)
[`Helsinki-NLP/opus-100`](https://huggingface.co/datasets/Helsinki-NLP/opus-100), configuration `en-ur` — downloaded automatically in the notebook with `load_dataset("Helsinki-NLP/opus-100", "en-ur")`. No local file is needed.

## Pretrained models
- Urdu → English: [`Helsinki-NLP/opus-mt-ur-en`](https://huggingface.co/Helsinki-NLP/opus-mt-ur-en)
- English → Urdu: [`Helsinki-NLP/opus-mt-en-ur`](https://huggingface.co/Helsinki-NLP/opus-mt-en-ur)

## Fine-tuned models (my Hugging Face repos)
- `YOUR_HUGGINGFACE_USERNAME/urdu-english-translation`
- `YOUR_HUGGINGFACE_USERNAME/english-urdu-translation`

## Pipeline
1. Load + clean OPUS-100 (en-ur) → 2. Tokenize → 3. Fine-tune both directions (`Seq2SeqTrainer`) → 4. Evaluate with BLEU (sacreBLEU) → 5. Push to Hugging Face → 6. Streamlit app.

The full workflow is in `Urdu_English_NLP_Translation_Project.ipynb` (run on Google Colab with a T4 GPU).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io → **New app** → select your repo, branch `main`, file `app.py`.
3. In **Advanced settings → Secrets** add:
   ```toml
   HF_USERNAME = "your-huggingface-username"
   ```
4. Click **Deploy**. If your fine-tuned repos are not found, the app automatically uses the pretrained base models.

## Results
Add your BLEU scores here after running the notebook:
- Urdu → English BLEU: _fill in_
- English → Urdu BLEU: _fill in_
