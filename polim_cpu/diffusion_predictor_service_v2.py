"""Change ONLY predict; formal phase2 correct is imported unchanged."""
import json
from pathlib import Path
import sys
from . import large_step_transport_service_v1 as service
from .large_step_phase2_service_v1 import correct
from .diffusion_predictor_v2 import predict


def predictor(adapter,request,output):
    h,diagnostics,fields=predict(request['thickness_old'],request['bed'],request['smb_requested'],
        float(request['dt_years']),adapter.rates(request['velocity_old']))
    service._write(output,thickness=h,**fields)
    output.with_suffix('.diagnostics.json').write_text(json.dumps(diagnostics,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    service._predict=predictor;service._correct=correct
    print('{"ready":true}',flush=True)
    raise SystemExit(service.serve(Path(sys.argv[1])))
