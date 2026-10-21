from pathlib import Path
import sys
project=Path(__file__).resolve().parent
code=(project/'yx_derive_incremental_delivery_v3.py').read_text(encoding='utf-8')
code=code.replace("old=Path('H:/ToYangxin/result_yx/completed_reconstruction_20261009_v2_650')","old=Path('H:/ToYangxin/result_yx/completed_reconstruction_20261009_v3_651')")
code=code.replace("assert len(current)==651 and set(current.index)-set(prior.index)=={'YX-04312'}","assert len(prior)==651 and set(prior.index).issubset(current.index)\nnew_ids=sorted(set(current.index)-set(prior.index))\nassert new_ids and set(new_ids).issubset({'YX-10304','YX-21852','YX-11977'})")
code=code.replace('==651,table','==len(current),table')
code=code.replace("'old650-derived-products-reused-one-new04312-derived'","'prior651-derived-products-reused-final-supplements-only-derived'")
code=code.replace("'old650_thickness_and_common_surface_change_rasters_byte_copied':True","'old651_thickness_and_common_surface_change_rasters_byte_copied':True")
code=code.replace("'old650_bed_alignment_stats_reused':True","'old651_bed_alignment_stats_reused':True")
code=code.replace("'new_derivative_records':['YX-04312']","'new_derivative_records':new_ids")
exec(compile(code,str(project/'yx_derive_final_incremental_v4.py'),'exec'),{'__name__':'__main__','__file__':str(project/'yx_derive_final_incremental_v4.py')})
