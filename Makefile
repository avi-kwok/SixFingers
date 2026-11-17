PY := .venv/bin/python

.PHONY: setup data features train eval serve test

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
