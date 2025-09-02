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

stats:
	python analysis/03_descriptives_pct.py

annotate_gpt2:
	python scripts/annotate_file.py data/raw/gpt2/gpt2_generated.csv analysis/annotated_gpt2.csv

annotate_bold:
	python scripts/annotate_file.py data/raw/bold/bold_selection.csv analysis/annotated_bold.csv

# per-subcorpus tables

# per-subcorpus tables

# per-subcorpus tables

all:
	make annotate_clauses
	make tables
	make stats
	make chisq

# per-subcorpus tables
tables_gpt2:
	IN=analysis/annotated_gpt2.csv python analysis/02_export_tables.py
tables_bold:
	IN=analysis/annotated_bold.csv python analysis/02_export_tables.py

# per-subcorpus percentages
stats_gpt2:
	IN=analysis/annotated_gpt2.csv python analysis/03_descriptives_pct.py
stats_bold:
	IN=analysis/annotated_bold.csv python analysis/03_descriptives_pct.py

# per-subcorpus chi-square
chisq_gpt2:
	IN=analysis/annotated_gpt2.csv python analysis/04_chisq_tests.py
chisq_bold:
	IN=analysis/annotated_bold.csv python analysis/04_chisq_tests.py

all_gpt2:
	make annotate_gpt2
	make tables_gpt2
	make stats_gpt2
	make chisq_gpt2

all_bold:
	make annotate_bold
	make tables_bold
	make stats_bold
	make chisq_bold
