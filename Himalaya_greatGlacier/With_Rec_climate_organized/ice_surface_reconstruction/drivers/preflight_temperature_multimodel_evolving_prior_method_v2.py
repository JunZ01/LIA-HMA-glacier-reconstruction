"""Version 2 of the multimodel evolving-prior method preflight.

V1 assumed every archived bivariate proxy stored its M_t and M_p components.
Eight non-tree records store only the combined HXb. V2 audits closure wherever
components are present and records the component-free count explicitly.
"""

from __future__ import annotations

from collections import Counter

import h5py
import numpy as np

import preflight_temperature_multimodel_evolving_prior_method_v1 as v1


v1.OUTPUT = v1.LOOP / "temperature_multimodel_evolving_prior_method_preflight_v2"
v1.DRIVER = __file__


def proxy_inventory(path):
    methods: Counter[str] = Counter()
    maximum_closure = 0.0
    all_component_lengths: set[int] = set()
    component_proxy_count = 0
    component_free_proxy_count = 0
    with h5py.File(path, "r") as archive:
        refs = np.asarray(archive["proxy_data/full_proxy"]).reshape(-1, order="F")
        for ref in refs:
            proxy = archive[ref]
            method = v1.decode_char(proxy["psm_mthd"])
            methods[method] += 1
            hxb = np.asarray(proxy["HXb"], dtype=float).reshape(-1, order="F")
            if not all(name in proxy for name in ("M_t", "M_p")):
                component_free_proxy_count += 1
                continue
            b = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
            mt = np.asarray(proxy["M_t"], dtype=float).reshape(-1, order="F")
            mp = np.asarray(proxy["M_p"], dtype=float).reshape(-1, order="F")
            if b.size != 3 or not (mt.size == mp.size == hxb.size):
                raise RuntimeError("Unexpected archived bivariate PSM structure")
            rebuilt = b[0] + b[1] * mt + b[2] * mp
            maximum_closure = max(maximum_closure, float(np.max(np.abs(rebuilt - hxb))))
            all_component_lengths.add(int(hxb.size))
            component_proxy_count += 1
    return {
        "proxy_count": int(sum(methods.values())),
        "psm_method_counts": dict(methods),
        "all_psms_require_paired_temperature_and_precipitation": bool(
            set(methods).issubset({"bi", "bivariate"})
        ),
        "stored_component_proxy_count": component_proxy_count,
        "component_free_proxy_count": component_free_proxy_count,
        "component_member_lengths": sorted(all_component_lengths),
        "maximum_hxb_rebuild_error": maximum_closure,
    }


v1.proxy_inventory = proxy_inventory


if __name__ == "__main__":
    raise SystemExit(v1.main())
