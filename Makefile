.PHONY: all annotate_clauses post tables stats chisq audit

all: annotate_clauses post tables stats chisq audit

annotate_clauses:
\tpython scripts/annotate_clauses.py

post:
\tpython analysis/00_postprocess_annotations.py --input analysis/annotated_pilot_clauses.csv --output analysis/annotated_pilot_clauses.clean.csv

tables:
\tpython analysis/02_export_tables.py --input analysis/annotated_pilot_clauses.clean.csv

stats:
\tpython analysis/03_descriptives_pct.py --input analysis/annotated_pilot_clauses.clean.csv

chisq:
\tpython analysis/04_chisq_tests.py

audit:
\tpython analysis/06_error_audit.py --input analysis/annotated_pilot_clauses.clean.csv --n 30 --seed 42
