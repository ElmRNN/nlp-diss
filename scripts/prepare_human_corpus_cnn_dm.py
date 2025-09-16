# Build a ~7,500-word human NEWS subcorpus from CNN/DailyMail.
# Writes: data/raw/human_cnn_dm.csv with columns: doc_id, subcorpus, focal_gender, text

import os, re, random
from datetime import datetime
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import spacy

# ---- CONFIG ----
TARGET_WORDS = 7500
OUT_CSV = "data/raw/human_cnn_dm.csv"
SUBCORPUS_NAME = "human"
SPLIT = "train"           # 'train'|'validation'|'test'
VERSION = "3.0.0"         # REQUIRED by HF now
MAX_ARTICLES = 2000
SEED = 42
MIN_W, MAX_W = 6, 80      # sentence word-length bounds
RE_HE  = re.compile(r"\bhe\b",  re.I)
RE_SHE = re.compile(r"\bshe\b", re.I)
# ----------------

def wc(s: str) -> int:
    return len((s or "").strip().split())

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    print(f"[load] cnn_dailymail {VERSION} / split={SPLIT} …")
    ds = load_dataset("cnn_dailymail", VERSION, split=SPLIT)  # <- single, versioned call

    df = ds.to_pandas()[["article"]].dropna()
    df = df.sample(frac=1.0, random_state=SEED).head(MAX_ARTICLES).reset_index(drop=True)
    print(f"[info] articles sampled: {len(df)}")

    print("[nlp] loading spaCy sentencizer …")
    nlp = spacy.load("en_core_web_sm", disable=["ner","tagger","lemmatizer"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    fem_rows, male_rows = [], []
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

    # Deduplicate and shuffle
    fem_rows = list(dict.fromkeys(fem_rows))
    male_rows = list(dict.fromkeys(male_rows))
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
        out.append({"doc_id": f"H{i:05d}", "subcorpus": SUBCORPUS_NAME, "focal_gender": "female", "text": s}); i += 1
    for s in male_keep:
        out.append({"doc_id": f"H{i:05d}", "subcorpus": SUBCORPUS_NAME, "focal_gender": "male", "text": s}); i += 1

    random.seed(SEED)
    random.shuffle(out)

    # Snapshot + final write
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    raw_out = OUT_CSV.replace(".csv", f".raw.{ts}.csv")
    pd.DataFrame(out).to_csv(raw_out, index=False, encoding="utf-8")
    df_out = pd.DataFrame(out).drop_duplicates(subset=["text"]).reset_index(drop=True)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    approx_words = int(df_out["text"].str.split().str.len().sum())
    print(f"[done] {OUT_CSV} | rows: {len(df_out)} | approx words: {approx_words} | "
          f"genders: {df_out['focal_gender'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()
