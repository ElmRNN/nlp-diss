## Corpus Design
- Subcorpora: GPT-2 prompt-controlled; BOLD news-like items.
- Target size: ~60–80 texts per subcorpus (pilot first).
- Inclusion criteria: gendered referent clearly identifiable; news-like tone.

## Annotation
- Unit: clause (clauselets via rule split; later: refinement).
- Features: role, voice, passive agent presence, process type, modality, hedge.
- Tools: spaCy en_core_web_sm + rule heuristics; manual overrides for flags.

## Reliability
- Pilot IAA on ~10 texts: compute percent agreement + Cohen's κ on role/voice.
- Revise codebook after disagreements; lock scheme.

## Statistics
- Binary features: χ² (or Fisher when sparse); effect sizes (Cramér’s V).
- Multi-category (process type): χ² with adjusted residuals.

## Reproducibility
- Repo structure; requirements.txt; seeds/params in YAML; scripts produce tables/figures from raw.
