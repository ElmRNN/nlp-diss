.PHONY: pilot annotate_clauses descriptives tables
pilot:
\tpython scripts/annotate_spacy.py

annotate_clauses:
\tpython scripts/annotate_clauses.py

descriptives:
\tpython analysis/01_descriptives.py

tables:
\tpython analysis/02_export_tables.py
