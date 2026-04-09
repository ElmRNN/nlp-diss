import pandas as pd
import scipy.stats as stats
import numpy as np
from pathlib import Path

IN_CSV = "analysis/annotated_gpt35_clauses.csv"
OUT_DIR = Path("analysis/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MD = OUT_DIR / "gpt35_passivisation.md"
OUT_CSV = OUT_DIR / "gpt35_passivisation.csv"


def cramers_v(chi2, n, r, c):
    return 0.0 if n == 0 or min(r - 1, c - 1) <= 0 else float(np.sqrt(chi2 / (n * (min(r - 1, c - 1)))))


df = pd.read_csv(IN_CSV)

# keep only female/male rows
df = df[df["focal_gender"].isin(["female", "male"])].copy()

# keep only active/passive rows for passivisation
df = df[df["voice"].isin(["active", "passive"])].copy()

# contingency table
tbl = pd.crosstab(df["focal_gender"], df["voice"])

# force column order
for col in ["active", "passive"]:
    if col not in tbl.columns:
        tbl[col] = 0
tbl = tbl[["active", "passive"]]

# row percentages
pct = tbl.div(tbl.sum(axis=1), axis=0) * 100
pct = pct.round(1)

# chi-square
n = int(tbl.values.sum())
if tbl.shape[0] < 2 or tbl.shape[1] < 2 or n == 0:
    chi2, p, dof, v = 0.0, 1.0, 0, 0.0
else:
    chi2, p, dof, _ = stats.chi2_contingency(tbl)
    v = cramers_v(chi2, n, *tbl.shape)

# markdown output
lines = ["# GPT-3.5 Passivisation Analysis\n"]
lines.append("## passivisation counts")
lines.append(tbl.to_markdown())
lines.append(f"\n## passivisation percentages by gender")
lines.append(pct.to_markdown())
lines.append(f"\nχ²={chi2:.2f}, df={dof}, p={p:.4f}, Cramér’s V={v:.3f}\n")

OUT_MD.write_text("\n".join(lines), encoding="utf-8")

with open(OUT_CSV, "w", encoding="utf-8") as f:
    f.write("# passivisation_counts\n")
    tbl.to_csv(f)
    f.write(f"\n# passivisation_percentages\n")
    pct.to_csv(f)
    f.write(f"\n# chi2={chi2}, df={dof}, p={p}, V={v}\n")

print("PASSIVISATION COUNTS")
print(tbl)
print("\nPASSIVISATION PERCENTAGES")
print(pct)
print(f"\nχ²={chi2:.2f}, df={dof}, p={p:.4f}, Cramér’s V={v:.3f}")
print(f"\nWrote:\n - {OUT_MD}\n - {OUT_CSV}")
