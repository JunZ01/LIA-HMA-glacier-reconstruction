"""Phase-1 transport equations with auditable final-state fluxes and constraints."""
import sys
from pathlib import Path
import numpy as np
from . import large_step_transport_service_v1 as service
from .large_step_transport_v1 import implicit_conservative_step


def correct(adapter, request, output):
    old, h, smb = (request[k] for k in ('thickness_old', 'candidate_thickness', 'smb_requested'))
    dt = float(request['dt_years'])
    rates = adapter.rates(request['velocity'])
    fixed = implicit_conservative_step(old, smb, dt, rates)
    area = adapter.topology.dual_area
    donor, _, rate = rates.donor_receiver_rate()
    flux = rates.internal_signed * h[donor]
    boundary = rates.boundary_outward * h[adapter.topology.boundary_donors]
    unmet = fixed.unmet_ablation_volume
    residual = fixed.operator @ h - fixed.rhs_requested - unmet
    v0 = float(area @ old)
    source = float(dt * (area @ fixed.actual_smb))
    export = float(dt * boundary.sum())
    scale = max(v0 + abs(source) + dt * (np.abs(flux).sum() + boundary.sum()), 1.)
    fields = dict(thickness=fixed.thickness, actual_smb=fixed.actual_smb,
                  unmet_ablation_volume=unmet, internal_flux=flux, boundary_flux=boundary,
                  local_residual=residual, boundary_export_volume=export,
                  actual_source_volume=source, potential_source_volume=float(dt * (area @ smb)),
                  mass_relative=abs(float(area @ h) - v0 - source + export)/max(v0,abs(source),abs(export),1.),
                  thickness_equation_relative=float(np.abs(residual).sum()/scale),
                  fixed_point_relative=float(np.sum(area*np.abs(fixed.thickness-h))/max(v0,1.)),
                  active_iterations=fixed.active_iterations,
                  active_set_equal=np.array_equal(h == 0, fixed.thickness == 0),
                  local_equation_scale=scale)
    service._write(output, **{k:np.asarray(v) for k,v in fields.items()})


if __name__ == '__main__':
    service._correct = correct
    print('{"ready": true}', flush=True)
    raise SystemExit(service.serve(Path(sys.argv[1])))
