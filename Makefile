PY := .venv/bin/python

.PHONY: setup data features train eval serve test smoke full

setup:
	python3.11 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

data:
	$(PY) -m ml.data.build --limit $(or $(LIMIT),200)

features:
	$(PY) -m ml.features.extract --limit $(or $(LIMIT),200)

# manual promote: training alone never moves models/current
train:
	$(PY) -m ml.probes.train --promote

eval:
	$(PY) -m ml.eval.run

serve:
	$(PY) -m uvicorn api.main:app --host 127.0.0.1 --port 8000

test:
	$(PY) -m pytest tests -q

# smoke scale
smoke:
	$(PY) -m ml.data.build --limit 50 --skip-download
	test -f data/manifest.parquet
	$(PY) -m ml.features.extract --limit 200
	$(PY) -m ml.probes.train --promote
	$(PY) -m ml.eval.run
	$(PY) tests/smoke_api.py
	cd web && npm run build

# full scale, hours; every stage is resumable, so rerun `make full` after an interruption; logs in logs/
FULL_LIMIT ?= 2500
full:
	mkdir -p logs
	@echo "[full] download (log: logs/full-download.log)"
	$(PY) -m ml.data.download --limit $(FULL_LIMIT) >> logs/full-download.log 2>&1
	@echo "[full] generate (log: logs/full-generate.log)"
	$(PY) -m ml.data.generate_local --limit $(FULL_LIMIT) >> logs/full-generate.log 2>&1
	@echo "[full] build (log: logs/full-build.log)"
	$(PY) -m ml.data.build --limit $(FULL_LIMIT) --skip-download --freeze >> logs/full-build.log 2>&1
	@echo "[full] features (log: logs/full-features.log)"
	$(PY) -m ml.features.extract --limit 0 >> logs/full-features.log 2>&1
	@echo "[full] train (log: logs/full-train.log)"
	$(PY) -m ml.probes.train --promote >> logs/full-train.log 2>&1
	@echo "[full] eval (log: logs/full-eval.log)"
	$(PY) -m ml.eval.run >> logs/full-eval.log 2>&1
