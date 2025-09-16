import argparse
import pandas as pd
from pathlib import Path

def pct(n, d): return 0.0 if d==0 else 100*n/d

def compute(df):
    rows=[]
    for g in sorted(df["focal_gender"].dropna().unique()):
        dfg=df[df["focal_gender"]==g]
        n_cl=len(dfg)
        n_sub=(dfg["role"]=="subject").sum()
        n_passive=(dfg["voice"]=="passive").sum()
        n_passive_sub=((dfg["role"]=="subject") & (dfg["voice"]=="passive")).sum()
        n_agent_omit=((dfg["voice"]=="passive") & (dfg["agent_present"]=="no")).sum()
        n_modal=(dfg["modality"].astype(str).str.lower()=="yes").sum()
        n_hedge=(dfg["hedge"].astype(str).str.lower()=="yes").sum()
        rows.append({
            "gender":g,"n_clauses":n_cl,"n_subject":int(n_sub),
            "% subject":round(pct(n_sub,n_cl),1),
            "% passive (of subjects)":round(pct(n_passive_sub,max(n_sub,1)),1),
            "% agent omitted (of passives)":round(pct(n_agent_omit,max(n_passive,1)),1),
            "% modality":round(pct(n_modal,n_cl),1),
            "% hedge":round(pct(n_hedge,n_cl),1),
        })
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    args=ap.parse_args()
    df=pd.read_csv(args.input)

    outdir=Path("analysis/tables"); outdir.mkdir(parents=True,exist_ok=True)

    overall=compute(df)
    overall.to_csv(outdir/"pilot_descriptives_by_gender.csv",index=False)
    overall.to_markdown(outdir/"pilot_descriptives_by_gender.md",index=False)

    cond=compute(df[df["role"]!="absent"])
    cond.to_csv(outdir/"pilot_descriptives_by_gender_cond_presence.csv",index=False)
    cond.to_markdown(outdir/"pilot_descriptives_by_gender_cond_presence.md",index=False)

    if "subcorpus" in df.columns:
        subs=[]
        for sub, dsub in df.groupby("subcorpus"):
            tmp=compute(dsub); tmp.insert(0,"subcorpus",sub); subs.append(tmp)
        if subs:
            subdf=pd.concat(subs, ignore_index=True)
            subdf.to_csv(outdir/"pilot_descriptives_by_gender_subcorpus.csv",index=False)
            subdf.to_markdown(outdir/"pilot_descriptives_by_gender_subcorpus.md",index=False)

if __name__=="__main__":
    main()
