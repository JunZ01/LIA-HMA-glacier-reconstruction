"""Independent GPU production entry point; default is a no-model static check."""

from lia_gpu_production_v1.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
