PY := .venv/bin/python

.PHONY: setup test

setup:
	python3.11 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

test:
	$(PY) -m pytest tests -q
