PACKAGE_NAME := shrimpy

.PHONY: install
install:
	pip install -e ".[dev]"

.PHONY: uninstall
uninstall:
	pip uninstall -y $(PACKAGE_NAME)

.PHONY: check
check:
	ruff format --check .
	ruff check .

.PHONY: format
format:
	ruff format .
	ruff check --fix .

.PHONY: test
test:
	python -m pytest
