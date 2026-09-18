# Contributing guide

Thanks for your interest in contributing to `shrimPy`!

## Getting started

Please read the [README](./README.md) for an overview of the project,
and how you can install and use the package. Formatting and linting are handled
by [ruff](https://docs.astral.sh/ruff/), which replaces black, isort and flake8.

## Making changes

Any change made to the `main` branch or release maintenance branches
need to be proposed in a [pull request](https://github.com/czbiohub/shrimPy/pulls) (PR).

Follow [these instructions](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
to [fork](https://github.com/czbiohub/shrimPy/fork) the repository.

## Setting up a development environment

1. Install the package and the development tooling:

```sh
uv sync
```

`dev` is a [dependency group](https://peps.python.org/pep-0735/), not an extra, so it is installed by default and cannot be requested as `[dev]`. Optional runtime features are extras instead: `uv sync --extra gui`, `--extra viewer`, `--extra dynatrack`.

2. Install pre-commit hooks:

```sh
pre-commit install
```

The pre-commit hooks automatically run style checks when staged changes are committed. Resolve any violations before committing your changes. You can run them over the whole tree at any time with `pre-commit run --all-files`, or run the underlying checks directly with `make check`.

## Makefile

A [makefile](Makefile) is included to help with a few basic development commands. Currently, the following commands are available:

```sh
make install     # install the package and dev tooling (uv sync)
make install-dev # additionally overlay editable installs of sibling
                 # pymmcore-plus / ome-writers checkouts, for co-development
make uninstall   # uninstall the package
make check       # check formatting and linting
make format      # apply formatting and linting changes
make test        # run the test suite
```

## Testing

See [docs/testing.md](docs/testing.md) for how to run the suite. Most tests need
no hardware; the demo-device integration tests require the Micro-Manager
adapters, which on Linux must be built from source (steps and HPC-specific notes
are in that doc).
