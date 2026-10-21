# IITM Homogeneous Indian Temperature Qualification v2 Contract

## Purpose

Acquire and qualify the official IITM Homogeneous Indian Monthly Surface
Temperature product for the Western Himalaya and Northeast India regions as
external evidence for the HMA 1850/1860-1950 residual-temperature diagnosis.
This is a read-only evidence experiment. It does not authorize a temperature
correction, production reconstruction, EnKF run, or OGGM run.

## Version boundary

An interactive format probe preceding this contract displayed official rows
for 1921-1925. No displayed value was saved, calculated, or used. That probe
is invalid and excluded from the evidence chain. Version 2 starts from a new
result directory and is the only admissible experiment version.

## Frozen sources

Only files served by the IITM official `tropmet.res.in` domain are admissible:

- data archive page `static_pages.php?page_id=54`;
- `README.pdf`;
- `NEW-TNREGION.TXT` (regional Tmin);
- `NEW-TXREGION.TXT` (regional Tmax);
- `TEMP-REG.jpg` (official homogeneous-region map).

The primary methodological reference is Kothawale and Rupa Kumar (2005),
doi:10.1029/2005GL023528. Publisher metadata or article text may be used for
methodological interpretation, but no third-party numerical temperature file
may enter the experiment.

## Immutable-source rule

Downloaded files are written only to a new `raw_unopened` directory, hashed
with SHA-256, and made read-only. Analysis writes derived files elsewhere.
Hashes are checked again after analysis.

## Locked-period parser contract

For every candidate data row the parser must:

1. match only the row prefix and decode region code, year, and variable tag;
2. reject 1921-1939 immediately;
3. reject every year outside 1901-1920 and 1940-1950 immediately;
4. only then slice, tokenize, and convert the twelve monthly values.

The synthetic test places nonnumeric poison tokens in 1921-1939. The test
passes only if those tokens never reach the numeric converter. Derived tables
must contain only 1901-1920 and 1940-1950.

## Variables and seasons

- IITM-published variables: monthly Tmin and Tmax in degrees Celsius.
- Derived diagnostic only: `Tmean_derived = (Tmax + Tmin) / 2`.
- Annual: January-December.
- Winter: previous December plus current January-February (DJF), computed only
  when all three months occur in the same allowed segment.
- Pre-monsoon: March-May (MAM).
- Monsoon: June-September (JJAS).
- Post-monsoon: October-November (ON), matching the 2005 paper definition.

No average may cross the sealed 1921-1939 gap.

## Qualification boundaries

- Western Himalaya is relevant regional instrumental evidence for the western
  Indian Himalaya, not a full western-HMA grid validation.
- Northeast India is an eastern HMA-margin regional diagnostic, not a Tibetan
  Plateau or full eastern-HMA grid validation.
- The series are secondary products made from IMD station observations,
  objective 0.5-degree interpolation, a 1951-1980 climatology, and regional
  averaging. `Homogeneous` names the climate regions; absent explicit evidence
  of station breakpoint homogenization, the product must not be described as
  independently homogenized station data.
- IITM, GHCN, and CRU may share underlying IMD observations. IITM is admitted
  as an independent processing and regionalization route, not an independent
  observational vote against those products.

## Frozen outputs

- official source manifest and post-run integrity check;
- locked-access audit and poison-test result;
- allowed monthly series and annual/seasonal aggregates;
- coverage table;
- linear trend/direction precheck for 1901-1920 and 1940-1950;
- official-method and source-dependence audit;
- a figure containing only allowed values;
- `FINAL_RESULT.md` with a qualification decision.

Trend estimates over 20 and 11 years are descriptive prechecks, not correction
coefficients or decisive climate trends. No acceptance decision may be tuned
after seeing their values.
