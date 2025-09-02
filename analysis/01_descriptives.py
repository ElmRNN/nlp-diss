import pandas as pd
IN_CSV = "analysis/annotated_pilot.csv"
df = pd.read_csv(IN_CSV)

def pct(n, d):
   return 0 if d==0 else 100*n/d
summary = []
for g in df["focal_gender"].unique():
    dfg = df[df["focal_gender"]==g]
    subj = (dfg["role"]=="subject").sum()
    passive = (dfg["voice"]=="passive").sum()
    subj_rate = pct(subj, len(dfg))
    passive_rate = pct(passive, subj)  # of subjects
    agent_omit = pct(((dfg["voice"]=="passive") & (dfg["agent_present"]=="no")).sum(), passive)
    summary.append({
        "gender": g, "n_rows": len(dfg),
        "% subject": round(subj_rate,1),
        "% passive (of subjects)": round(passive_rate,1),
        "% agent omitted (of passives)": round(agent_omit,1),
    })

print(pd.DataFrame(summary).to_string(index=False))
