import re
import pandas as pd
from pathlib import Path

IN_CSV = "data/recovered/hf_generations.csv"
OUT_CSV = "analysis/gpt2_for_annotation.csv"


def detect_gender_and_term(title: str):
    t = str(title).lower()

    female_terms = ["woman", "girl", "female", "mother", "wife"]
    male_terms = ["man", "boy", "male", "father", "husband"]

    for term in female_terms:
        if re.search(r"\b" + re.escape(term) + r"\b", t):
            return "female", term

    for term in male_terms:
        if re.search(r"\b" + re.escape(term) + r"\b", t):
            return "male", term

    return "unknown", ""


def main():
    df = pd.read_csv(IN_CSV)

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()

    df["text"] = df["generated_text"].astype(str)

    detected = df["prompt_title"].apply(detect_gender_and_term)
    df["focal_gender"] = detected.apply(lambda x: x[0])
    df["focal_term"] = detected.apply(lambda x: x[1])

    df["subcorpus"] = "gpt2"

    out = df[
        [
            "pair_id",
            "doc_id",
            "condition",
            "prompt_title",
            "model_name",
            "subcorpus",
            "focal_gender",
            "focal_term",
            "text",
        ]
    ].copy()

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"Wrote {len(out)} rows to {OUT_CSV}")
    print(out["focal_gender"].value_counts(dropna=False))


if __name__ == "__main__":
    main()