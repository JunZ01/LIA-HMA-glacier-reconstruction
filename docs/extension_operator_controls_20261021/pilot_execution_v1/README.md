# Modern volume-tau pilot: completed; progression stopped

The unchanged gates pass 12/22 selected systems (54.545%), below the fixed 80%
progression rule. Do not launch the proposed 191-system rerun from this pilot.
All 22 selected systems remain in the denominator: 17 have volume roots within
40-260 kPa and 5 do not; among the 17 solved cases, 4 exceed the allowed profile
RMSE ratio and 1 exceeds the allowed surface-RMSE increase. Failure IDs and reasons
are explicit in pilot_results.csv and pilot_summary.json. Thresholds were not changed.

This run used original float64 NPZs, not the transfer float32 files. Nineteen systems
were calculated locally, with two single-thread workers; the three existing supplied
controls were reused, retaining their external provenance. Elapsed local batch time
was 293.693 seconds. Every new system saved its row/curve immediately in an immutable
fingerprinted checkpoint. The controller is archived here; all per-system records
remain in the canonical local evidence. No completed fit was rerun for reporting.

The supplied 27-system balanced design yielded only 22 under its unmodified selector:
West 9, Central 9, East 4. The unrepresented multi-member population is 156/191.
The result therefore applies to this singleton pilot, not a validated population-wide
performance estimate. Modern closure is calibration rather than independent validation.
No LIA, climate or dynamic run, production promotion, or accepted-ledger change occurred.
Project facts1501/1502 record authorization/completion; both state validations passed.
