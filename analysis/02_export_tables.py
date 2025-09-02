import os, pandas as pd

IN_CSV = os.environ.get("IN", "analysis/annotated_pilot_clauses.csv")
if "gpt2" in IN_CSV:
    OUT_MD = "analysis/tables/gpt2_role_voice.md"
elif "bold" in IN_CSV:
    OUT_MD = "analysis/tables/bold_role_voice.md"
else:
    OUT_MD = "analysis/tables/pilot_role_voice.md"

df = pd.read_csv(IN_CSV)
tbl = (df
  .groupby(["focal_gender","role","voice","agent_present"], dropna=False)
  .size()
  .reset_index(name="n")
  .sort_values(["focal_gender","role","voice","agent_present"])
)

def to_md(d):
    lines = ["| focal_gender | role | voice | agent_present | n |",
             "|---|---|---|---|---|"]
    for r in d.itertuples(index=False):
        lines.append(f"| {r.focal_gender} | {r.role} | {r.voice} | {r.agent_present} | {r.n} |")
    return "\n".join(lines)

md = to_md(tbl)
with open(OUT_MD, "w") as f:
    f.write(md+"\n")

print(md)
print(f"\nWrote {OUT_MD}")
