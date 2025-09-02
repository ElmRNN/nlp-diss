import os, pandas as pd, scipy.stats as stats, numpy as np
from pathlib import Path
IN_CSV = os.environ.get("IN", "analysis/annotated_pilot_clauses.csv")
label = "pilot"
if "gpt2" in IN_CSV: label = "gpt2"
elif "bold" in IN_CSV: label = "bold"

OUT_DIR = Path("analysis/tables"); OUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(IN_CSV)

def cramers_v(chi2, n, r, c):
    return 0.0 if n==0 or min(r-1,c-1)<=0 else float(np.sqrt(chi2 / (n*(min(r-1,c-1)))))

def chisq(feature, by="focal_gender"):
    tbl = pd.crosstab(df[by], df[feature].fillna("NA"))
    n = int(tbl.values.sum())
    if tbl.shape[0]<2 or tbl.shape[1]<2 or n==0:
        return (feature, tbl, {"chi2":0.0,"df":0,"p":1.0,"cramerv":0.0})
    chi2, p, dof, _ = stats.chi2_contingency(tbl)
    return (feature, tbl, {"chi2":chi2,"df":dof,"p":p,"cramerv":cramers_v(chi2,n,*tbl.shape)})

tests = ["role","voice","agent_present","process_type","modality","hedge"]
results = [chisq(feat) for feat in tests]

md_path = OUT_DIR / f"{label}_chisq_tests.md"
csv_path = OUT_DIR / f"{label}_chisq_tests.csv"

lines = [f"# χ² Tests by Gender — {label}\n"]
for name, tbl, res in results:
    lines.append(f"## {name}")
    try:
        lines.append(tbl.to_markdown())
    except Exception:
        lines.append(tbl.to_csv())
    lines.append(f"\nχ²={res['chi2']:.2f}, df={res['df']}, p={res['p']:.4f}, Cramér’s V={res['cramerv']:.3f}\n")

md_path.write_text("\n".join(lines), encoding="utf-8")
with open(csv_path, "w", encoding="utf-8") as f:
    for name, tbl, res in results:
        f.write(f"# {name}\n"); tbl.to_csv(f); f.write(f"# chi2={res['chi2']}, df={res['df']}, p={res['p']}, V={res['cramerv']}\n\n")

print(f"Wrote:\n - {md_path}\n - {csv_path}")
