import pandas as pd
from pathlib import Path

IN_CSV = "analysis/annotated_pilot_clauses.csv"
OUT_DIR = Path("analysis/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN_CSV)

def pct(n, d):
    return 0.0 if d == 0 else 100.0 * n / d

rows = []
for g in sorted(df["focal_gender"].dropna().unique()):
    dfg = df[df["focal_gender"] == g]
    n_all = len(dfg)

    n_subject = (dfg["role"] == "subject").sum()
    n_passive = ((dfg["role"] == "subject") & (dfg["voice"] == "passive")).sum()
    n_passives_agent_no = ((dfg["voice"] == "passive") & (dfg["agent_present"] == "no")).sum()
    n_modality = (dfg["modality"] == "yes").sum()
    n_hedge = (dfg["hedge"] == "yes").sum()

    rows.append({
        "gender": g,
        "n_clauses": n_all,
        "n_subject": n_subject,
        "% subject": round(pct(n_subject, n_all), 1),
        "% passive (of subjects)": round(pct(n_passive, n_subject), 1),
        "% agent omitted (of passives)": round(pct(n_passives_agent_no, n_passive), 1),
        "% modality": round(pct(n_modality, n_all), 1),
        "% hedge": round(pct(n_hedge, n_all), 1),
    })

out = pd.DataFrame(rows)

# Save CSV
csv_path = OUT_DIR / "pilot_descriptives_by_gender.csv"
out.to_csv(csv_path, index=False)

# Save Markdown
md_lines = [
    "| gender | n_clauses | n_subject | % subject | % passive (of subjects) | % agent omitted (of passives) | % modality | % hedge |",
    "|---|---:|---:|---:|---:|---:|---:|---:|"
]
for r in out.to_dict(orient="records"):
    md_lines.append(
        f"| {r['gender']} | {r['n_clauses']} | {r['n_subject']} | {r['% subject']} | {r['% passive (of subjects)']} | {r['% agent omitted (of passives)']} | {r['% modality']} | {r['% hedge']} |"
    )

md_path = OUT_DIR / "pilot_descriptives_by_gender.md"
md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

# Print to console
print(out.to_string(index=False))
print(f"\nWrote:\n - {csv_path}\n - {md_path}")
