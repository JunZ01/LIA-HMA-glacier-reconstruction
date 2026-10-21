from __future__ import annotations

import h5py
import numpy as np
import scipy.io

import audit_enkf_reconstruction_provenance_v2 as compatibility


audit = compatibility.audit


def _verify_lineage(archive_root, data_root):
    combined_path = data_root / "ENKF_Three_ensemble50.mat"
    export_path = data_root / "5rec_result" / "tas" / "tas_result_matlab_for_py.mat"
    export_struct = scipy.io.loadmat(
        export_path,
        variable_names=["export_data_for_python"],
        struct_as_record=True,
        squeeze_me=False,
    )["export_data_for_python"][0, 0]
    rows = []
    result_root = archive_root / "fixed_proxy_experiments_69" / "tas" / "results"
    for dataset_id in (6, 9):
        for method, file_name in audit.METHOD_FILE_NAMES.items():
            result_var = f"{method}_results_{dataset_id}"
            summary_var = f"{method}_{dataset_id}"
            combined = scipy.io.loadmat(
                combined_path,
                variable_names=[result_var, summary_var],
                struct_as_record=True,
                squeeze_me=False,
            )
            result_struct = combined[result_var][0, 0]
            summary_struct = combined[summary_var][0, 0]
            combined_members = np.asarray(result_struct["Xa_m_all"], dtype=float)
            combined_mean = np.asarray(summary_struct["Xa_m_all"], dtype=float)
            combined_status = np.asarray(result_struct["experiment_status"]).reshape(-1)
            fixed_path = result_root / f"proxy{dataset_id}" / file_name
            with h5py.File(fixed_path, "r") as handle:
                fixed_members_hdf = np.asarray(handle["Xa_m_all"], dtype=float)
                fixed_status = np.asarray(handle["experiment_status"]).reshape(-1)
            fixed_members_matlab = np.transpose(fixed_members_hdf, (2, 1, 0))
            fixed_mean_matlab = np.nanmean(fixed_members_hdf, axis=0).T
            export_field = f"{method}_{dataset_id}_Xa_m"
            exported_data = np.asarray(export_struct[export_field][0, 0]["data"], dtype=float)
            exported_as_state = exported_data.reshape((336, 1001), order="C")
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "method": method,
                    "fixed_to_combined_members_max_abs_diff": float(
                        np.nanmax(np.abs(fixed_members_matlab - combined_members))
                    ),
                    "fixed_mean_to_combined_summary_max_abs_diff": float(
                        np.nanmax(np.abs(fixed_mean_matlab - combined_mean))
                    ),
                    "combined_summary_to_python_export_max_abs_diff": float(
                        np.nanmax(np.abs(exported_as_state - combined_mean))
                    ),
                    "fixed_and_combined_status_equal": bool(
                        np.array_equal(fixed_status, combined_status)
                    ),
                    "export_reshape_order": "C",
                    "fixed_member_shape_matlab": list(fixed_members_matlab.shape),
                    "combined_member_shape": list(combined_members.shape),
                    "combined_summary_shape": list(combined_mean.shape),
                    "exported_data_shape": list(exported_data.shape),
                }
            )
    return rows


audit._verify_lineage = _verify_lineage


if __name__ == "__main__":
    audit.main()
