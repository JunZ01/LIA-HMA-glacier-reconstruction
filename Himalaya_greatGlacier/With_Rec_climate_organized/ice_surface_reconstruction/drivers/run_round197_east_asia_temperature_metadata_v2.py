from __future__ import annotations

import json as std_json

import run_round197_east_asia_temperature_metadata_v1 as v1


class JsonApiCompatibility:
    @staticmethod
    def loads(value: str):
        record = std_json.loads(value)
        for item in record.get("files", []):
            links = item.get("links", {})
            if "content" not in links and "self" in links:
                links["content"] = links["self"]
        return record

    @staticmethod
    def dumps(*args, **kwargs):
        return std_json.dumps(*args, **kwargs)


v1.OUT = v1.ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2/"
    "round_197_east_asia_temperature_metadata_v2"
)
v1.json = JsonApiCompatibility


if __name__ == "__main__":
    v1.main()
