import argparse, random
import pandas as pd
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--n",type=int,default=30)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()

    df=pd.read_csv(args.input)
    focus=df[(df["role"]!="absent")|(df["voice"]=="passive")|
             (df["modality"].astype(str).str.lower()=="yes")|
             (df["hedge"].astype(str).str.lower()=="yes")]
    if focus.empty: focus=df
    sample=focus.sample(n=min(args.n,len(focus)),random_state=args.seed)
    outdir=Path("analysis/tables"); outdir.mkdir(parents=True,exist_ok=True)
    sample.to_csv(outdir/"pilot_error_audit_sample.csv",index=False)
    (outdir/"pilot_error_audit_sample.md").write_text(sample.to_markdown(index=False))
    print("Wrote audit sample")

if __name__=="__main__":
    main()
