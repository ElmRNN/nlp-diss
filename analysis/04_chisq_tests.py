cat > analysis/04_chisq_tests.py << 'PY'
import pandas as pd
import scipy.stats as stats
import numpy as np
from pathlib import Path

IN_CSV = "analysis/annotated_pilot_clauses.csv"
OUT_DIR = Path("analysis/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN_CSV)

def cramers_v(chi2, n, r, c):
    """Cramér's V effect size for chi² tables."""
    return 0.0 if n == 0 or min(r-1, c-1) <= 0 else float(np.sqrt(chi2 / (n * (min(r-1, c-1)))))

def chisq_test(feature, by="focal_gender"):
    # Build contingency table; drop NaNs in the feature to avoid shape issues
    tbl = pd.crosstab(df[by], df[feature].fillna("NA"))
    n = int(tbl.values.sum())
    if tbl.shape[0] < 2 or tbl.shape[1] < 2 or n == 0:
        # Not enough categories to test
        return (feature, tbl, {"chi2": 0.0, "df": 0, "p": 1.0, "cramerv": 0.0})
    chi2, p, dof, expected = stats.chi2_contingency(tbl)
    v = cramers_v(chi2, n, *tbl.shape)
    return (feature, tbl, {"chi2": chi2, "df": dof, "p": p, "cramerv": v})

# ---- Run a set of tests ----
tests_to_run = [
    "role",
    "voice",
    "agent_present",
    "process_type",
    "modality",
    "hedge",
]

results = [chisq_test(feat, by="focal_gender") for feat in tests_to_run]

# --- Output with safe fallback ---
md_path = OUT_DIR / "pilot_chisq_tests.md"
csv_path = OUT_DIR / "pilot_chisq_tests.csv"

lines = ["# χ² Tests by Gender\n"]
for name, tbl, res in results:
    lines.append(f"## {name}")
    try:
        # Pretty Markdown table (requires 'tabulate' via 'pip install tabulate')
        lines.append(tbl.to_markdown())
    except Exception:
        # Fallback to CSV-style string if Markdown export fails
        lines.append(tbl.to_csv())
    lines.append(f"\nχ²={res['chi2']:.2f}, df={res['df']}, p={res['p']:.4f}, Cramér’s V={res['cramerv']:.3f}\n")

md_path.write_text("\n".join(lines), encoding="utf-8")

# Always also write a raw CSV bundle of the tables + stats
with open(csv_path, "w", encoding="utf-8") as f:
    for name, tbl, res in results:
        f.write(f"# {name}\n")
        tbl.to_csv(f)
        f.write(f"# chi2={res['chi2']}, df={res['df']}, p={res['p']}, V={res['cramerv']}\n\n")

print(f"Wrote:\n - {md_path}\n - {csv_path}")
PY


