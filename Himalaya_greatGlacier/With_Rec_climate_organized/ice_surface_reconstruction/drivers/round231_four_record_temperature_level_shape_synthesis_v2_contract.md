# Round 231 v2 Contract

The scientific design, inputs, and decision gate are unchanged from Round 231
v1. Version 2 corrects one schema assumption: the Round 227 lineage decision
is stored at `evidence.decision`, not at the JSON root. A derived bridge copy
adds the same value at the root, records the authoritative source path and
SHA256, and is written only inside the independent v2 output directory.

