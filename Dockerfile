# CPU image for the CLI (built and smoke-tested in CI).
#   docker build -t creditdecide .
#   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" creditdecide \
#       run --dataset credit-default --out results/credit --plot
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLCONFIGDIR=/tmp/mpl

WORKDIR /src
COPY pyproject.toml README.md LICENSE DATA_LICENSE.md ./
COPY src ./src
RUN pip install ".[plot]" && rm -rf /src

RUN useradd --create-home app
USER app
WORKDIR /work
ENTRYPOINT ["creditdecide"]
CMD ["--help"]
