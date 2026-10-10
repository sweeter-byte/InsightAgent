# InsightAgent Chapter 18 Demo Notes

The project codename for the repeatable Chapter 18 demo is **Harbor Lantern**.

The acceptance marker is **IA-CH18-READY**. A correct local-knowledge answer
must recover this marker from the indexed material rather than invent it from
general model knowledge.

Harbor Lantern keeps production HTTP serving and offline Evaluation separate:
the service handles interactive queries and research runs, while the top-level
`evals` package reads versioned datasets and recorded fixtures.
