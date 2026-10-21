# Round 156 Duan 2019 Temperature Identity Contract

## Question

Is the public Duan et al. (2019) whole-Tibetan-Plateau August-September
temperature reconstruction numerically distinct from the archived proxy6 and
proxy9 records, and what scientific role is admissible before any trajectory
comparison?

## Frozen Design

- Keep the NOAA download byte-identical and read it only through a new output.
- Parse the year token first. Skip every 1921-1939 row before splitting or
  converting any value token.
- Retain only 1000-1920 and 1940-2000 in the allowed export.
- Compare only the published primary `Temp` column with every archived proxy6
  and proxy9 sequence using the established Round 149 exact-affine and
  near-exact-standardized thresholds.
- Record metadata for the uncertainty columns but do not treat them as separate
  proxy candidates.
- Do not inspect the candidate's 1901-1920 trend or compare it with the project
  reconstruction in this round.

## Decision Boundary

Zero sequence aliases do not prove source independence because the candidate is
a derived reconstruction from a 17-site MXD network. This round may release it
only to a later source-material audit and modern seasonal stability diagnostic.
It cannot be assimilated as a raw proxy, used as independent final validation,
or authorize a paleo reconstruction, EnKF production run, or OGGM run.
