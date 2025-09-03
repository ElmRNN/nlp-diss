import pandas as pd, os
def summarize(path, label):
    df = pd.read_csv(path)
    out=[]
    for g in sorted(df["focal_gender"].dropna().unique()):
        d = df[df["focal_gender"]==g]
        n = len(d)
        n_subj = (d["role"]=="subject").sum()
        n_pass = ((d["role"]=="subject") & (d["voice"]=="passive")).sum()
        n_pass_agent_no = ((d["voice"]=="passive") & (d["agent_present"]=="no")).sum()
        n_mod = (d["modality"]=="yes").sum()
        n_hed = (d["hedge"]=="yes").sum()
        pct = lambda a,b: 0.0 if b==0 else 100*a/b
        out.append({
            "subcorpus": label, "gender": g, "n": n,
            "% subject": round(pct(n_subj,n),1),
            "% passive (of subj)": round(pct(n_pass,n_subj),1),
            "% agent omitted (of passives)": round(pct(n_pass_agent_no,n_pass),1),
            "% modality": round(pct(n_mod,n),1),
            "% hedge": round(pct(n_hed,n),1),
        })
    return pd.DataFrame(out)

rows=[]
paths = [
    ("pilot", "analysis/annotated_pilot_clauses.csv"),
    ("gpt2",  "analysis/annotated_gpt2.csv"),
    ("bold",  "analysis/annotated_bold.csv"),
]
for label, path in paths:
    if os.path.exists(path):
        rows.append(summarize(path, label))
df = pd.concat(rows, ignore_index=True)
df = df[["subcorpus","gender","n","% subject","% passive (of subj)","% agent omitted (of passives)","% modality","% hedge"]]
md = df.to_markdown(index=False)

out_md = "analysis/tables/compare_pilot_gpt2_bold.md"
with open(out_md,"w") as f: f.write(md+"\n")
print(md); print(f"\nWrote {out_md}")
