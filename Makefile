PACKAGE_NAME := shrimpy

.PHONY: install
install:
	uv sync

# Optional: only for co-developing pymmcore-plus / ome-writers. `uv sync` alone
# fetches both from the git branches pinned in pyproject.toml; this overlays
# editable installs from sibling checkouts in the parent directory.
.PHONY: install-dev
install-dev:
	uv sync
	uv pip install -e ../pymmcore-plus -e ../ome-writers

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
