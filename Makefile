.PHONY: pilot annotate_clauses descriptives tables

pilot:
	python scripts/annotate_spacy.py

annotate_clauses:
	python scripts/annotate_clauses.py

descriptives:
	python analysis/01_descriptives.py

tables:
	python analysis/02_export_tables.py

chisq:
	python analysis/04_chisq_tests.py
