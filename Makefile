PY := .venv/bin/python

.PHONY: setup data features train test

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

test:
	$(PY) -m pytest tests -q
