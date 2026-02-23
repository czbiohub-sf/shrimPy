PACKAGE_NAME := shrimpy

.PHONY: install
install:
	uv sync

.PHONY: uninstall
uninstall:
	uv pip uninstall $(PACKAGE_NAME)

.PHONY: check
check:
	uv run ruff format --check .
	uv run ruff check .

.PHONY: format
format:
	uv run ruff format .
	uv run ruff check --fix .

.PHONY: test
test:
	uv run pytest
