PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python

.PHONY: setup test demo clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

test:
	PYTHONPATH=src $(VENV_PYTHON) -m pytest -q --capture=no

demo:
	PYTHONPATH=src $(VENV_PYTHON) scripts/download_data.py
	PYTHONPATH=src $(VENV_PYTHON) scripts/run_demo.py

clean:
	rm -rf .pytest_cache reports/*.md $(VENV)
