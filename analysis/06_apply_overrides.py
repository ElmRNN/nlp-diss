import pandas as pd

IN="analysis/annotated_pilot_clauses.csv"
OV="analysis/overrides.csv"
OUT="analysis/annotated_pilot_clauses_corrected.csv"

df = pd.read_csv(IN)
try:
    ov = pd.read_csv(OV, comment="#")
except FileNotFoundError:
    ov = pd.DataFrame(columns=["doc_id","sent_id","clause_id","column","value"])

for row in ov.itertuples(index=False):
    mask = (df.doc_id==row.doc_id) & (df.sent_id==row.sent_id) & (df.clause_id==row.clause_id)
    if row.column in df.columns:
        df.loc[mask, row.column] = row.value

df.to_csv(OUT, index=False)
print(f"Wrote {OUT}")
