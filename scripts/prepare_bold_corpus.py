# File: scripts/prepare_bold_corpus.py
"""
Balanced human (BOLD) subcorpus matching pipeline format.

It will:
- try to load the BOLD "news" domain from Hugging Face Datasets
- extract a text field (best-effort: tries 'text', 'content', 'sentence', etc.)
- filter rows that contain singular 'he' or 'she' (case-insensitive, word boundaries)
- balance MALE/FEMALE by **total words** (not just rows) toward TARGET_WORDS
- deduplicate, drop too-short/too-long lines
- write data/raw/bold_selection.csv with columns:
    doc_id,subcorpus,focal_gender,text


Requirements:
    pip install datasets pandas tqdm
"""

import os
import re
import sys
from math import inf
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# CONFIG 
TARGET_WORDS = 7500                 # ~7.5k words for the human subcorpus
OUT_CSV = "data/raw/bold_selection.csv"
SUBCORPUS_NAME = "bold"
SEED = 42

# acceptable length bounds (words) per row
MIN_WORDS = 6
MAX_WORDS = 80

# regex for singular he/she (not part of another word)
RE_HE = re.compile(r"\bhe\b", flags=re.IGNORECASE)
RE_SHE = re.compile(r"\bshe\b", flags=re.IGNORECASE)

# candidate dataset ids/configs to try in order
CANDIDATES = [
    # (dataset_id, config)
    ("BOLD", "news"),
    ("HuggingFaceH4/BOLD", "news"),
    ("bigscience/BOLD", "news"),
    ("BOLD", None),
    ("HuggingFaceH4/BOLD", None),
    ("bigscience/BOLD", None),
]

# candidate text columns to look for in the dataset
TEXT_COLUMNS = ["text", "content", "sentence", "body", "document", "article"]
# ---


def word_count(s: str) -> int:
    return len(s.strip().split())


def install_hint(msg: str):
    print("\n[!] " + msg)
    print("    If missing, install with:   pip install datasets pandas tqdm\n")


def try_load_bold():
    """
    Try several dataset ids/configs; return a pandas DataFrame with a text column.
    Raises RuntimeError if none worked.
    """
    try:
        from datasets import load_dataset  # noqa
    except Exception as e:
        raise RuntimeError(
            "Hugging Face 'datasets' is not installed."
        ) from e

    last_err = None
    for ds_id, cfg in CANDIDATES:
        try:
            if cfg:
                dset = load_dataset(ds_id, cfg)
            else:
                dset = load_dataset(ds_id)
            # pick a split heuristically
            split_name = "train" if "train" in dset else list(dset.keys())[0]
            ds = dset[split_name]
            df = ds.to_pandas()
            # find a reasonable text column
            text_col = None
            for c in TEXT_COLUMNS:
                if c in df.columns:
                    text_col = c
                    break
            if text_col is None:
                # fallback: pick the first string-like column with many non-null values
                for c in df.columns:
                    if df[c].dtype == object and df[c].notna().sum() > 100:
                        text_col = c
                        break
            if text_col is None:
                raise ValueError(f"No plausible text column found in columns={list(df.columns)}")
            df = df[[text_col]].rename(columns={text_col: "text"})
            # drop nulls/empties
            df = df[df["text"].astype(str).str.strip().ne("")]
            print(f"[ok] Loaded dataset '{ds_id}' config='{cfg}' with {len(df)} rows; using column '{text_col}'")
            return df
        except Exception as e:
            last_err = e
            print(f"[info] Could not use dataset '{ds_id}' config='{cfg}': {e}")
            continue
    raise RuntimeError(f"Failed to load any BOLD dataset candidate. Last error: {last_err}")


def filter_gender_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure str
    df = df.copy()
    df["text"] = df["text"].astype(str)

    # length filter first (quick)
    df["wc"] = df["text"].map(word_count)
    df = df[(df["wc"] >= MIN_WORDS) & (df["wc"] <= MAX_WORDS)]

    # he/she filters (singular via word boundaries)
    female = df[df["text"].str.contains(RE_SHE)]
    male = df[df["text"].str.contains(RE_HE)]

    # remove any overlaps (rare but possible if both words appear)
    both_idx = set(female.index).intersection(set(male.index))
    if both_idx:
        female = female.drop(index=both_idx, errors="ignore")
        male = male.drop(index=both_idx, errors="ignore")

    # deduplicate text within each group
    female = female.drop_duplicates(subset=["text"])
    male = male.drop_duplicates(subset=["text"])

    print(f"[info] Candidate rows – female: {len(female)}, male: {len(male)}")
    return female, male


def cumulative_sample_by_words(df: pd.DataFrame, target_words: int, desc: str) -> pd.DataFrame:
    """
    Deterministically shuffle and take rows until we meet/exceed target_words.
    """
    if df.empty:
        return df
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    total = 0
    keep_idx = []
    for i, row in enumerate(df.itertuples()):
        total += row.wc
        keep_idx.append(i)
        if total >= target_words:
            break
    out = df.iloc[keep_idx].copy()
    print(f"[ok] Selected {len(out)} {desc} rows for ~{total} words (target {target_words})")
    return out


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    try:
        df = try_load_bold()
    except Exception as e:
        install_hint("Could not load BOLD dataset via Hugging Face.")
        print(f"Error detail: {e}")
        print("\nAs a fallback, place a local CSV at data/raw/bold_source.csv with a 'text' column, then re-run.")
        sys.exit(1)

    # Extract candidate male/female rows
    female, male = filter_gender_rows(df)

    if female.empty or male.empty:
        print("[!] Not enough he/she rows found in this split. Consider changing dataset/config or loosening filters.")
        sys.exit(1)

    # balance by total words toward TARGET_WORDS
    # split the target evenly between female/male
    half_target = TARGET_WORDS // 2
    female_sel = cumulative_sample_by_words(female, half_target, desc="female")
    male_sel = cumulative_sample_by_words(male, half_target, desc="male")

    # build final frame
    final = pd.concat([female_sel.assign(focal_gender="female"),
                       male_sel.assign(focal_gender="male")],
                      ignore_index=True)
    final = final.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # final tidy and ids
    final["text"] = final["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    final["doc_id"] = [f"B{i:05d}" for i in range(1, len(final) + 1)]
    final["subcorpus"] = SUBCORPUS_NAME
    out = final[["doc_id", "subcorpus", "focal_gender", "text"]].copy()

    # write
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    raw_out = OUT_CSV.replace(".csv", f".raw.{ts}.csv")
    out.to_csv(raw_out, index=False, encoding="utf-8")
    print(f"[ok] Raw balanced selection written: {raw_out}")

    # light duplicate removal across whole set
    out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[ok] Final CSV written: {OUT_CSV}")
    print(f"[done] Rows: {len(out)}, Approx words: {sum(len(t.split()) for t in out['text'])}")


if __name__ == "__main__":
    main()

