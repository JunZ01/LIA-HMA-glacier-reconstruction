"""Invoke the untouched canonical validator; return its full result as an artifact."""
import io,sys,os,json,contextlib,importlib.util,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    target=ROOT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py'
    out=Path(sys.argv[1]);buffer=io.StringIO();code=1;failure=None
    try:
        spec=importlib.util.spec_from_file_location('unchanged_lia_validator',target);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with contextlib.redirect_stdout(buffer):code=module.main()
    except BaseException:failure=traceback.format_exc()
    # This is an ordinary data file opened by the child, not its standard stream.
    with out.open('x',encoding='utf8') as f:f.write(buffer.getvalue());f.flush();os.fsync(f.fileno())
    if failure:
        with out.with_suffix('.exception').open('x',encoding='utf8') as f:f.write(failure);f.flush();os.fsync(f.fileno())
    return code
if __name__=='__main__':raise SystemExit(main())
