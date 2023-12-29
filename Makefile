setup:
	poetry install --with dev
	poetry run pre-commit install -t pre-commit -t pre-push -t commit-msg -t post-checkout -t post-merge
	git config --local core.commentchar ";"
	git config --local commit.template .gitmessage

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*

clean: clean-pyc clean-test

test: clean
	poetry run pytest test --cov=useful --cov-report term-missing --cov-report xml

test-prepush: clean
	poetry run pytest test

lint:
	poetry run ruff check .

format:
	poetry run ruff format .

deps:
	poetry run deptry .

check: test lint format deps
