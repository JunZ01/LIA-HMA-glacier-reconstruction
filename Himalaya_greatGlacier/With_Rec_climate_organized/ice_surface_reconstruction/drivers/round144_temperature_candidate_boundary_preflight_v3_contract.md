# Round 144 Temperature-Candidate Boundary Preflight V3

V2 correctly parsed the verified whitespace-delimited Tucson values and then stopped at a one-byte DOS end-of-file marker (`0x1A`) in the first source. V3 changes only the text-container parser to skip a line consisting solely of that verified terminal marker before applying the existing minimum-length check. The V2 tokenization rule and every V1 scientific, source, sealed-record, period, and value-boundary rule remain unchanged.
