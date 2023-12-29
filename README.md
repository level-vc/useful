# useful

![ci workflow badge](https://github.com/level-vc/useful/actions/workflows/ci.yml/badge.svg?branch=main)

![useful](/logo.png)

This package provides the python SDK for useful machines.

## Installation

Currently, you can run

```bash
poetry add useful-machines
```

```bash
pip install useful-machines
```

For the library to work successfully and to access the Useful tool, please enter the waitlist on usefulmachines.dev.

## Development

### First Time

0. We use `git` as our version control system and Github as our source of truth.
   Please install `git` and ensure you have access to the repository.

1. We assume you have a way of managing python versions, e.g., [`pyenv`](https://github.com/pyenv/pyenv).
   Please ensure that you have the python version located in `.python-version` installed.
   If you use `pyenv`, you can run `pyenv install` to install the correct version.

2. We use [`poetry`](https://python-poetry.org/) for dependency management.
   Please install it.

3. With those tools installed, please enter the project root and run:

```bash
make setup
```

### General

When working on this project, you can activate the virtual environment with:

```bash
poetry shell
```

If you use VS Code as your IDE, you can set the python interpreter to the one in the virtual environment, which will enable several tools.

### Tooling

We use a variety of tools to ensure code quality and for build tasks like documentation.
These types of tools are available in the `Makefile`; if you inspect that file, you can find the various commands available.
We also use [`pre-commit`](https://pre-commit.com/) to run some of these tools automatically while using `git`; we have selected a pragmatic configuration that runs quick checks every time you commit and more thorough checks when you push.

### Commit Process

We use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) for commit messages and require additional information within the body of a commit; please follow this format.
The `.gitmessage` file contains a template for commit messages that should be automatically picked up by `git` after running `make setup`.
To work on new code, please branch off of the `develop` branch and write code in a feature branch.
Then, open a pull request to merge into `develop`.

When merging the commit, ensure that:

-   you only have one commit; rebase and squash if necessary
-   your commit message has the proper format
-   your commit is signed
-   your commit passes all checks
-   your PR is approved

### Troubleshooting

If the tools are not running correctly, run `make setup` to set them up again.
