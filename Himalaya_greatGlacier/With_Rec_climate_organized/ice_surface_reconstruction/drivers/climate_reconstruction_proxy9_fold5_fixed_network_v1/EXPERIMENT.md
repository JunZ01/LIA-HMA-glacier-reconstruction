# Proxy9 Fold-5 Fixed-Network Control v1

This no-OGGM control separates the 1981-2000 fold-5 operator effect from the
declining number of available proxy observations.

Four fixed-input EnKF branches are compared:

- archived 1940-2000 operator with the original year-varying network;
- fold-5 1901-1980 operator with the original year-varying network;
- archived operator using only the 125 records present in every year from
  1981 through 2000;
- fold-5 operator using the same fixed 125-record network.

The common-network branches set all other proxy observations to missing. They
do not delete or alter source records. Prior, samples, localization, archived
RegEM error variances, PSM model type, HMA mask, and operators remain frozen.
The experiment diagnoses network composition; it is not a production network
selection.
