from pathlib import Path


root = Path(__file__).resolve().parent
files = [
    root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/run_lia_191_split_forward_delta_pilot_v1.py",
    root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/lia_forward_delta_from_saved_spinup.py",
]

for path in files:
    text = path.read_text(encoding="utf-8")
    if "def _source_task_path(" not in text:
        marker = "\n\ndef _task_option(task: dict, name: str) -> str | None:\n"
        if path.name.startswith("run_lia_191_split_forward_delta_pilot"):
            marker = "\n\ndef _audit_path(source: Path, summary: dict, task: dict) -> Path:\n"
        helper = (
            "\n\ndef _source_task_path(source: Path) -> Path:\n"
            "    for name in (\"task.json\", \"source_task.json\"):\n"
            "        candidate = source / name\n"
            "        if candidate.exists():\n"
            "            return candidate\n"
            "    raise FileNotFoundError(f\"missing-source-task-metadata:{source}\")\n"
        )
        if marker not in text:
            raise RuntimeError(f"helper-insertion-marker-not-found:{path}")
        text = text.replace(marker, helper + marker, 1)
    text = text.replace(
        'json.loads((source / "task.json").read_text(encoding="utf-8"))',
        'json.loads(_source_task_path(source).read_text(encoding="utf-8"))',
    )
    text = text.replace(
        'json.loads((source_dir / "task.json").read_text(encoding="utf-8"))',
        'json.loads(_source_task_path(source_dir).read_text(encoding="utf-8"))',
    )
    path.write_text(text, encoding="utf-8")
