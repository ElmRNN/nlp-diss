import argparse
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df["agent_present"] = df["agent_present"].fillna("n/a")

    grp = (df.groupby(["focal_gender","role","voice","agent_present"], dropna=False)
             .size().reset_index(name="n"))

    out_dir = Path("analysis/tables"); out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem
    csv_path = out_dir / f"{stem}_role_voice.csv"
    md_path  = out_dir / f"{stem}_role_voice.md"

    grp.to_csv(csv_path, index=False)
    md_path.write_text(grp.to_markdown(index=False))
    print(f"Wrote {csv_path} and {md_path}")

if __name__ == "__main__":
    main()
