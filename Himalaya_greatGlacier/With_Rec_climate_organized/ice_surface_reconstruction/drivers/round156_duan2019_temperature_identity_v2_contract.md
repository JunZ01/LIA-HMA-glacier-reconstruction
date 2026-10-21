# Round 156 Duan 2019 Temperature Identity V2 Contract

V1 stopped before any scientific comparison because the NOAA file declares
UTF-8 but contains Windows-1252 bytes. V2 changes only boundary-safe parsing:
read binary lines, extract the ASCII year first, skip 1921-1939 before decoding
the remainder, and decode only metadata plus allowed rows as Windows-1252.

All scientific questions, identity thresholds, candidate role limits, source
files, allowed years, and prohibitions in the V1 contract remain unchanged.
