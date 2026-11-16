PY := .venv/bin/python

.PHONY: setup data test

setup:
	python3.11 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

data:
	$(PY) -m ml.data.build --limit $(or $(LIMIT),200)

test:
	$(PY) -m pytest tests -q
