from __future__ import annotations

import numpy as np

import audit_enkf_reconstruction_provenance as audit


def _scalar_optional(group, name: str) -> float:
    if name not in group or group[name].size == 0:
        return np.nan
    return float(np.asarray(group[name]).reshape(-1)[0])


def _char_optional(group, name: str) -> str:
    if name not in group or group[name].size == 0:
        return ""
    return audit._decode_char(group[name])


def _dataset_size(group, name: str) -> int:
    return int(group[name].size) if name in group else 0


def _proxy_record(handle, reference, index: int):
    group = handle[reference]
    location = np.asarray(group["location"]).reshape(-1, order="F")
    coefficients = np.asarray(group["b"]).reshape(-1, order="F")
    data = np.asarray(group["data"]).reshape(-1, order="F")
    age_yng = np.asarray(group["age_yng"]).reshape(-1, order="F")
    finite = np.isfinite(data)
    finite_ages = age_yng[finite] if age_yng.size == data.size else np.array([])
    return {
        "proxy_index_1based": index + 1,
        "latitude": float(location[0]),
        "longitude": float(location[1]),
        "proxy_type": _char_optional(group, "ptype"),
        "psm_method": _char_optional(group, "psm_mthd"),
        "R": _scalar_optional(group, "R"),
        "pcr": _scalar_optional(group, "pcr"),
        "pcr_t": _scalar_optional(group, "pcr_t"),
        "pcr_p": _scalar_optional(group, "pcr_p"),
        "pval": _scalar_optional(group, "pval"),
        "b0": float(coefficients[0]) if coefficients.size > 0 else np.nan,
        "b1": float(coefficients[1]) if coefficients.size > 1 else np.nan,
        "b2": float(coefficients[2]) if coefficients.size > 2 else np.nan,
        "data_length": int(data.size),
        "data_finite_count": int(finite.sum()),
        "first_finite_age": float(np.nanmin(finite_ages)) if finite_ages.size else np.nan,
        "last_finite_age": float(np.nanmax(finite_ages)) if finite_ages.size else np.nan,
        "Rvec_length": _dataset_size(group, "Rvec"),
        "HXb_length": _dataset_size(group, "HXb"),
        "M_t_length": _dataset_size(group, "M_t"),
        "M_p_length": _dataset_size(group, "M_p"),
    }


audit._proxy_record = _proxy_record


if __name__ == "__main__":
    audit.main()
