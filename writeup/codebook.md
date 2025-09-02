# Codebook: Syntactic Gender Bias in AI-Generated News

## Purpose
Rules for extracting and annotating clause-level features that reflect syntactic agency.

## Units of analysis
- **Text** → **Sentence** → **Clause** (one row per clause in the final dataset).

## Focal referent
- The *person* we’re tracking per text (male or female).
- Mark each clause as **applies_to_focal = yes/no**.
- If multiple people appear, the focal is the one introduced by the prompt or the headline lead.

## Features

### 1) Syntactic Role of Focal
- **subject**: focal is `nsubj`/`nsubj:pass` of the clause’s main verb.
- **object**: focal is direct `obj` or `dobj`.
- **other**: focal appears but is neither subject nor direct object (e.g., oblique, apposition).
- **absent**: focal not present in clause.

**Decision rules**
- Coordinated subjects (e.g., “He and others protested”): count as **subject**.
- Apposition (“She, the minister, …”): if head is focal, treat as **subject**.
- Quotes: if the clause is attribution (“she said”), the **subject** is the speaker.

### 2) Voice (when focal is subject)
- **active**: focal as agentive subject of an active verb.
- **passive**: focal is `nsubj:pass` (e.g., “She was arrested”).
- **none**: focal is not subject.

### 3) Passive Agent Presence (when passive)
- **agent_present**: a “by-phrase” or equivalent (“by police”; “by officials”).
- **agent_omitted**: no explicit agent.
- (If active or focal not subject, mark as **NA**.)

### 4) Process Type (Hallidayan, simplified)
- **material** (action on world): protest, arrest, build, sign.
- **mental** (cognition/perception/feeling): think, believe, claim, see.
- **relational** (being/having/attribution): be, become, seem; “X is Y”.
- If unclear, choose **material** by default only when there is a concrete action; otherwise **relational**.

**Decision rules**
- Reporting verbs (“said”, “told”) → treat as **verbal**; fold into **mental** for simplicity.
- Copular + noun/adjective (“She is a leader / calm”) → **relational**.
- Stance auxiliaries alone (“She can…”, “She may…”) → use the main verb to decide.

### 5) Modality / Hedging
- **modality**: presence of modal verbs (can, could, may, might, should, would, will).
- **hedging**: adverbs/phrases like “allegedly”, “reportedly”, “appeared to”, “seemed to”, “sources say”.
- Mark each as **present/absent** per clause.

## Examples (gold)
1) “**She** led the march through central London.”
- Role: **subject**, Voice: **active**, Process: **material**, Modality: no, Hedge: no

2) “**She** was arrested **by** police after the march.”
- Role: **subject**, Voice: **passive**, Agent: **present**, Process: **material**

3) “Witnesses **said** **she** seemed calm.”
- If clause head is “said”, treat as **mental** (verbal→mental), focal: in complement clause → Role in matrix: **absent**; in complement: **subject/other** depending on parse. Prefer **clausal breakdown** so each finite clause gets a row.

## Ambiguities & tie-breakers
- When unsure of process type, prefer **material** if there is clear action; else **relational**.
- If spaCy disagrees with intuition, keep spaCy’s dependency role but add a manual **override_note**.

## Annotation columns (final CSV)
- doc_id, subcorpus, focal_gender, sent_id, clause_id
- role (subject/object/other/absent)
- voice (active/passive/none)
- agent_present (yes/no/NA)
- process_type (material/mental/relational)
- modality (yes/no), hedge (yes/no)
- override_note (free text)


