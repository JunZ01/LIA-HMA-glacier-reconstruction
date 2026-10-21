"""Create Round 163 v2 by controlled exact replacements after preflight."""

from pathlib import Path


driver = Path(__file__).resolve().parent
source = driver / "run_round163_temperature_prior_mode_psm_observability_v1.py"
target = driver / "run_round163_temperature_prior_mode_psm_observability_v2.py"
if target.exists():
    raise FileExistsError(target)
text = source.read_text(encoding="utf-8")
replacements = {
    'EXPERIMENT_ID = "round_163_temperature_prior_mode_psm_observability_v1"':
        'EXPERIMENT_ID = "round_163_temperature_prior_mode_psm_observability_v2"',
    'contract = driver / "round163_temperature_prior_mode_psm_observability_v1_contract.md"':
        'contract = driver / "round163_temperature_prior_mode_psm_observability_v2_contract.md"',
    '''        singular = np.asarray(basis["singular"][:rank], dtype=float)\n        u = np.asarray(basis["u"][:, :rank], dtype=float)\n        eigenvalues = singular**2 / (len(sample) - 1.0)\n        variance_fraction = eigenvalues / np.sum(eigenvalues)\n        scores = singular[:, None] * np.linalg.svd(\n            basis["root_weight"][:, None] * prior_anomaly[basis["selected"]], full_matrices=False\n        )[2][:rank]''':
        '''        weighted_prior = basis["root_weight"][:, None] * prior_anomaly[basis["selected"]]\n        u_full, singular_full, vt_full = np.linalg.svd(weighted_prior, full_matrices=False)\n        singular = np.asarray(singular_full[:rank], dtype=float)\n        u = np.asarray(u_full[:, :rank], dtype=float)\n        eigenvalues = singular**2 / (len(sample) - 1.0)\n        variance_fraction = eigenvalues / np.sum(eigenvalues)\n        scores = singular[:, None] * vt_full[:rank]''',
}
for old, new in replacements.items():
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"replacement-count:{count}:{old[:80]}")
    text = text.replace(old, new)
target.write_text(text, encoding="utf-8")
print(target)
