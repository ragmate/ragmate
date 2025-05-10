FROM python:3.13.2-slim

RUN apt update \
    \
    && apt install -y \
    gcc \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY ./requirements requirements
RUN pip install --no-cache-dir -r requirements/base.txt

COPY . /apps
WORKDIR /apps
