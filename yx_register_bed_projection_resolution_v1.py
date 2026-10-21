from pathlib import Path
import json
from yx_followup_transaction_v1 import register,sha
root=Path('H:/ToYangxin/result_yx/completion_followup_20261009_v1')
p=root/'remaining_failure_audit/bed_alignment_attribution/bounded_bed_alignment_attribution_summary.json'
assert p.exists()
out=root.parent/'completed_reconstruction_20261009_v2_650'
contract={'status':'fixed-native-source-bed-disparate-numerical-projection-resolved','evidence':{'path':str(p),'sha256':sha(p)},'diagnostic_csv':{'path':str(out/'common_grid_bed_alignment_diagnostics.csv'),'sha256':sha(out/'common_grid_bed_alignment_diagnostics.csv')},
 'independent_input_only_exact_transform_control':True,'PP_or_field_solver_called':False,'native_bed_or_preparation_normalization_changed':False,'original_model_arrays_rewritten':False,
 'source_field_definition':'fixed original raw-product native bed, not a guarantee that separately projected endpoint raster realizations are bit-identical',
 'derived_surface_change_policy':'Direct S_LIA minus S_modern on common cells with abs endpoint bed raster difference<=0.002m; other cells remain nodata; coverage recorded per system',
 'surface_rasters':650,'minimum_derived_common_coverage':.7791811627086169,'qualification_191_effect':0,
 'future_exact_projection_correction_requires_new_disclosed_mechanism':True}
print(register('yx_common_bed_projection_audit_v1',contract,
 'YX native fixed bed versus separately projected endpoint realizations: resolved diagnostic',
 'A new common-surface derivative check found that identical frozen native source beds did not yield bit-identical endpoint raster fields. Stop the affected derivative before publication and perform independent input-only attribution. Default GDAL0.125-source-pixel approximate transformation reproduces the extent-dependent differences; exact-transform controls eliminate common DEM differences in YX-00006,28179,25071 and all three source-edge classification flips in25071. Frozen native source hashes and preparation thickness-volume conservation pass; no endpoint renormalization or source bed change occurred. The fixed-bed statement refers to fixed original native bed inputs; separately warped numerical realizations require this qualification. Across650, median system bed-difference RMS is0m, maximum1.0225m, isolated maximum123.19m, summed absolute bed raster integral0.141762568km3. This integral is a diagnostic, not a nonlinear ice-volume error bound. Preserve all original models and their prior gates. Versioned common-S derivatives mask abs deltaB>0.002m, satisfy deltaS=deltaH+deltaB and record coverage; all650 have valid output and minimum coverage77.918%. No silent physical bed modification or accepted191 change. Any future exact-projection model rerun needs a new disclosed mechanism.',
 False,[],'Read-only release of the reconciled fixed-native-source interpretation and masked surface-change delivery; prospective model contracts must disclose the existing numerical projection assumption.',phase='resolved'))
