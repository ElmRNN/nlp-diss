import argparse
import pandas as pd
import spacy
from pathlib import Path
from lexicons import HEDGE_LEMMAS, HEDGE_PHRASES

def detect_hedge(text, nlp):
    s = (text or "").strip().lower()
    if not s:
        return "no"
    for ph in HEDGE_PHRASES:
        if ph in s:
            return "yes"
    doc = nlp(s)
    for t in doc:
        if t.lemma_.lower() in HEDGE_LEMMAS:
            return "yes"
    return "no"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="analysis/annotated_pilot_clauses.csv")
    ap.add_argument("--output", default="analysis/annotated_pilot_clauses.clean.csv")
    ap.add_argument("--spacy_model", default="en_core_web_sm")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # agent_present -> "n/a" if not passive; default "no" if passive but missing
    df["agent_present"] = df.apply(
        lambda r: "n/a" if str(r.get("voice","")).lower() != "passive" else (r.get("agent_present") or "no"),
        axis=1
    )

    nlp = spacy.load(args.spacy_model)
    df["hedge"] = df["text_span"].astype(str).map(lambda s: detect_hedge(s, nlp))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote cleaned annotations to {args.output} (n={len(df)})")

if __name__ == "__main__":
    main()
