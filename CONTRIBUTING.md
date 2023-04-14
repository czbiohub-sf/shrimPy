# Contributing guide

Thanks for your interest in contributing to `mantis`!

## Getting started

Please read the [README](./README.md) for an overview of the project,
and how you can install and use the package.

## Making changes

Any change made to the `main` branch or release maintenance branches
need to be proposed in a [pull request](https://github.com/czbiohub/mantis/pulls) (PR).

Follow [these instructions](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
to [fork](https://github.com/czbiohub/mantis/fork) the repository.

## Setting up a development environment

1. Install the package in development mode:

```
pip install ".[dev]"
```

2. Install pre-commit hooks:

```
pre-commit install
```

The pre-commit hooks automatically run style checks (e.g. `flake8`, `black`, `isort`) when staged changes are committed. Resolve any violations before committing your changes. You can manually run the pre-commit hooks at any time using the `make pre-commit` command.

## Makefile

A [makefile](Makefile) is included to help with a few basic development command. Currently, the following commands are available:

```sh
make setup-develop # setup the package in development mode
make uninstall # uninstall the package
make check-format # run black and isort format check
make format # run black and isort formatting
make lint # run flark8 linting
make pre-commit # run pre-commit hooks on all files
make test # run pytest
```