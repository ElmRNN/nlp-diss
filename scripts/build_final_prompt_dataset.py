import re
import pandas as pd
from pathlib import Path

INPUT = "data/recovered/articles_master.genderflip_pairs.final.csv"
OUTPUT_TITLES = "data/recovered/final_ai_prompts.csv"
OUTPUT_WRAPPED = "data/recovered/final_ai_prompts_wrapped.csv"


def clean_headline(title: str) -> str:
    t = str(title).strip()

    patterns = [
        r"\s*[-–|]\s*CNN(?:\.com)?\s*$",
        r"\s*[-–|]\s*Daily Mail\s*$",
        r"\s*[-–|]\s*MailOnline\s*$",
        r"\s*[-–|]\s*BBC(?:\s*News)?\s*$",
        r"\s*[-–|]\s*Reuters\s*$",
        r"\s*[-–|]\s*Associated Press\s*$",
        r"\s*[-–|]\s*AP\s*$",
        r"\s*\(\s*CNN(?:\.com)?\s*\)\s*$",
        r"\s*\(\s*Daily Mail\s*\)\s*$",
        r"\s*\(\s*MailOnline\s*\)\s*$",
        r"\s*\(\s*BBC(?:\s*News)?\s*\)\s*$",
        r"\s*\(\s*Reuters\s*\)\s*$",
        r"\s*[-–|]\s*[A-Za-z0-9.-]+\.(?:com|co\.uk|org|net)\s*$",
    ]

    for pattern in patterns:
        t = re.sub(pattern, "", t, flags=re.IGNORECASE)

    t = re.sub(r"\s+", " ", t).strip()
    return t


def wrap_prompt(clean_title: str) -> str:
    return (
        "Write a newspaper article in plain English based on the headline below.\n"
        "Write it as an original news report, not as a list, transcript, or commentary.\n"
        "Write at least 200 words.\n\n"
        f"Headline: {clean_title}\n\n"
        "Article:"
    )


def main():
    df = pd.read_csv(INPUT).copy()

    rows = []

    for _, row in df.iterrows():
        pair_id = int(row["pair_id"])
        doc_id = row["doc_id"]

        original_title = clean_headline(row["title_original"])
        flipped_title = clean_headline(row["title_flipped"])

        rows.append({
            "pair_id": pair_id,
            "doc_id": doc_id,
            "condition": "original",
            "prompt_title": original_title,
            "prompt_text": wrap_prompt(original_title),
        })

        rows.append({
            "pair_id": pair_id,
            "doc_id": doc_id,
            "condition": "flipped",
            "prompt_title": flipped_title,
            "prompt_text": wrap_prompt(flipped_title),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["pair_id", "condition"]).reset_index(drop=True)

    Path(OUTPUT_TITLES).parent.mkdir(parents=True, exist_ok=True)

    out[["pair_id", "doc_id", "condition", "prompt_title"]].to_csv(OUTPUT_TITLES, index=False)
    out.to_csv(OUTPUT_WRAPPED, index=False)

    print("Wrote:", OUTPUT_TITLES)
    print("Wrote:", OUTPUT_WRAPPED)
    print("Rows:", len(out))
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
