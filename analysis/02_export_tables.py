import pandas as pd

IN_CSV = "analysis/annotated_pilot_clauses.csv"
df = pd.read_csv(IN_CSV)

tbl = (df
  .groupby(["focal_gender","role","voice","agent_present"], dropna=False)
  .size()
  .reset_index(name="n")
  .sort_values(["focal_gender","role","voice","agent_present"])
)

def to_md(df):
    out = ["| focal_gender | role | voice | agent_present | n |",
           "|---|---|---|---|---|"]
    for r in df.itertuples(index=False):
        out.append(f"| {r.focal_gender} | {r.role} | {r.voice} | {r.agent_present} | {r.n} |")
    return "\n".join(out)

md = to_md(tbl)
with open("analysis/tables/pilot_role_voice.md","w") as f:
    f.write(md+"\n")
print(md)
