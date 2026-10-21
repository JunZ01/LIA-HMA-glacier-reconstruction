## 304. Central-HMA 2026 MXD public-data recovery closure (2026-08-21)

- Round 168 anonymously recovered the official supplementary PDF for Nahtz et al. (2026), DOI `10.1007/s11676-026-02104-0`. The unchanged 510,549-byte source has SHA256 `2000fe8de959bb0cb0349fe2a34e1ce9c323113158218bb1c53a0d185e5086e4`.
- The 12-page supplement contains figures S1-S18 and one calibration-statistics table, with no embedded file or linked CSV/TXT/XLS/ZIP/NetCDF annual dataset. No annual series was exported, OCR-processed, or digitized.
- Because the network begins in 1775, it cannot satisfy the Round 166 requirement that a new production-fitting candidate reach at least 1500. It remains metadata-only external evidence; anonymous public annual-data recovery is closed unless a new deposit appears.
- No locked-period numeric value, paleo trajectory, EnKF, or OGGM run was used. Evidence is in `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_168_central_hma_mxd_public_data_recovery_v1/`.

## 305. Huang 2025 public numeric-data recovery closure (2026-08-21)

- Round 169 recovered the official Elsevier supplementary DOCX for Huang et al. (2025), DOI `10.1016/j.quascirev.2025.109266`, from the anonymous `mmc1` attachment. The unchanged 995,460-byte OOXML source has SHA256 `bcd98acf8e876b5f4c09cdc10afb584a7dfba4202a6a10644e319696b9b1d787`.
- Structural inspection found 23 OOXML members, zero tables, zero embedded data members, and eight media images. No `mmc2` public spreadsheet exists at the standard Elsevier attachment path, and no exact public repository deposit was recovered.
- The public supplement therefore does not expose the annual 1241-2008 MXD chronology, the annual 1246-2008 reconstructed August-September minimum temperature, or raw sample measurements. No figure was rendered, OCR-processed, or digitized and no annual value was exported.
- Decision: close anonymous public numeric recovery. Retain Huang 2025 as a high-priority source-identity candidate because it passes the temporal and physical metadata gates, but do not inspect its paleo trajectory or use it as fitting/validation evidence without numeric data and source identity.
- No locked-period numeric value, EnKF, or OGGM run was used. Evidence is in `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_169_huang2025_mxd_public_data_recovery_v1/`.

## 306. Huang 2025 source-identity metadata boundary (2026-08-21)

- Round 170 parsed only safe keyword-matched OOXML paragraphs, core properties, relationships, and binary image metadata from the unchanged Round 169 DOCX. Ten safe paragraphs were exported; no locked paragraph occurred. Eight image members were hashed and dimensioned without rendering, OCR, or digitization.
- The supplement identifies a Ranwu MXD chronology calibrated against Bomin, Nyingchi, and combined Nyingchi-Bomi temperature records over 1961-2008. It calls the final product `our MXD chronology` but supplies no sample identifiers, site coordinates, accession numbers, or explicit statement of overlap with earlier Ranwu, Sygera, or Duan records.
- The Huang chronology begins in 1241, 134 years before the longest series reported for the Duan 17-site network begins in 1375. The Huang chronology therefore cannot be a complete numerical copy formed solely from that full Duan network, but partial site/sample overlap remains possible and scientific independence is unresolved.
- The official supplement exposes the corresponding-author mail link `zhuhf@itpcas.ac.cn`. A precise, unsent request for the annual chronology, reconstruction, raw measurements, coordinates, calibration equation, source crosswalk, and reuse license is archived in Round 170.
- Decision: `source_identity_unresolved_request_raw_data_and_crosswalk`. Do not promote Huang 2025, assume independence, or reopen the locked period from metadata alone. No annual value, paleo trajectory, EnKF, or OGGM run was used. Evidence is in `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_170_huang2025_source_identity_metadata_v1/`.
