PACKAGE_NAME := mantis

.PHONY: setup-develop
setup-develop:
	pip install -e ".[dev]"

.PHONY: uninstall
uninstall:
	pip uninstall -y $(PACKAGE_NAME)

.PHONY: check-format
check-format:
	black --check -S -t py39 .

.PHONY: format
format:
	black -S -t py39 .

.PHONY: lint
lint:
	flake8 $(PACKAGE_NAME)

# run the pre-commit hooks on all files (not just staged changes)
# (requires pre-commit to be installed)
.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files

.PHONY: test
test:
	python -m pytest . --disable-pytest-warnings
