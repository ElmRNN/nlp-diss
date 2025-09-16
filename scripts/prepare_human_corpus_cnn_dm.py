# File: scripts/prepare_human_corpus_cnn_dm.py
"""
Build a human news subcorpus from Hugging Face `ccdv/cnn_dailymail`:
- loads article texts written by journalists (CNN/DailyMail)
- sentence-splits with spaCy
- keeps sentences containing singular 'he' or 'she'
- balances female/male by TOTAL WORDS toward TARGET_WORDS
- writes data/raw/human_cnn_dm.csv with: doc_id,subcorpus,focal_gender,text

Requires:
  pip install datasets pandas tqdm spacy
  python -m spacy download en_core_web_sm
"""

import os, re
from datetime import datetime
from datasets import load_dataset  # pip install datasets
import pandas as pd
from tqdm import tqdm
import spacy

# CONFIG
TARGET_WORDS = 7500               # ~half of 15k total
OUT_CSV = "data/raw/human_cnn_dm.csv"
SUBCORPUS_NAME = "human"
SEED = 42

SPLIT = "train"                   # cnn_dailymail has 'train', 'validation', 'test'
VERSION = "3.0.0"                 # current common version on HF
MAX_ARTICLES = 2000               # safety cap
MIN_W = 6                         # sentence word-length bounds
MAX_W = 80
RE_HE  = re.compile(r"\bhe\b",  re.I)
RE_SHE = re.compile(r"\bshe\b", re.I)
# ---------------------------------------

def wc(s: str) -> int:
    return len(s.strip().split())

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    print("[load] Hugging Face cnn_dailymail …")
    try:
        ds = load_dataset("cnn_dailymail", VERSION, split=SPLIT)
    except Exception:
        ds = load_dataset("cnn_dailymail", split=SPLIT)  # fallback to latest default

    ds = load_dataset("cnn_dailymail", split=SPLIT)  # article field name: 'article'
    # Convert to pandas just to sample deterministically
    df = ds.to_pandas()[["article"]].dropna()
    df = df.sample(frac=1.0, random_state=SEED).head(MAX_ARTICLES).reset_index(drop=True)
    print(f"[info] articles sampled: {len(df)}")

    print("[nlp] loading spaCy en_core_web_sm for sentence splitting …")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    nlp.add_pipe("sentencizer") if "sentencizer" not in nlp.pipe_names else None

    # Collect candidate sentences with gendered pronouns
    fem_rows, male_rows = [], []
    total_processed = 0

    for art in tqdm(df["article"], desc="segmenting"):
        doc = nlp(art)
        for sent in doc.sents:
            txt = " ".join(sent.text.split())
            w = wc(txt)
            if w < MIN_W or w > MAX_W:
                continue
            has_she = bool(RE_SHE.search(txt))
            has_he  = bool(RE_HE.search(txt))
            if has_she and not has_he:
                fem_rows.append(txt)
            elif has_he and not has_she:
                male_rows.append(txt)
        total_processed += 1

    # Deduplicate
    fem_rows = list(dict.fromkeys(fem_rows))
    male_rows = list(dict.fromkeys(male_rows))
    print(f"[info] candidate sentences — female: {len(fem_rows)}, male: {len(male_rows)}")

    # Balance by total words toward TARGET_WORDS/2 each
    import random
    random.seed(SEED)
    random.shuffle(fem_rows)
    random.shuffle(male_rows)

    def take_until_words(rows, target):
        keep, total = [], 0
        for s in rows:
            total += wc(s)
            keep.append(s)
            if total >= target:
                break
        return keep, total

    half = TARGET_WORDS // 2
    fem_keep, fem_words = take_until_words(fem_rows, half)
    male_keep, male_words = take_until_words(male_rows, half)
    print(f"[ok] selected ~{fem_words} female words in {len(fem_keep)} sents; "
          f"~{male_words} male words in {len(male_keep)} sents.")

    # Assemble CSV
    out = []
    i = 1
    for s in fem_keep:
        out.append({"doc_id": f"H{i:05d}", "subcorpus": SUBCORPUS_NAME, "focal_gender": "female", "text": s})
        i += 1
    for s in male_keep:
        out.append({"doc_id": f"H{i:05d}", "subcorpus": SUBCORPUS_NAME, "focal_gender": "male", "text": s})
        i += 1

    # Shuffle final
    import random
    random.seed(SEED)
    random.shuffle(out)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    raw_out = OUT_CSV.replace(".csv", f".raw.{ts}.csv")
    pd.DataFrame(out).to_csv(raw_out, index=False, encoding="utf-8")
    print(f"[write] raw snapshot: {raw_out}")

    # Drop exact duplicates again just in case
    df_out = pd.DataFrame(out).drop_duplicates(subset=["text"]).reset_index(drop=True)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[done] final CSV: {OUT_CSV}")
    print("Rows:", len(df_out), "Approx words:", df_out["text"].str.split().str.len().sum())

if __name__ == "__main__":
    main()

