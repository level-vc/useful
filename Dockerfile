# syntax=docker/dockerfile:1
FROM python:3.10.6-slim-buster

WORKDIR /app

RUN pip install poetry
COPY pyproject.toml .
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction \
  && poetry install --with dev --no-interaction

COPY . .