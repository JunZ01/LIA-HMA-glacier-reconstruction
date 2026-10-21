# Round 197 V2 I/O correction

V1 stopped after writing only the Zenodo metadata because the current API uses
`files[].links.self` rather than `files[].links.content`. V2 changes only this
API-link compatibility detail and writes to a new independent result directory.
All scientific boundaries and admission rules in the V1 contract are frozen.

