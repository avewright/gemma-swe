# Sandbox image for trace collection: the contest sandbox (data/docker/Dockerfile.sandbox),
# plus what setup and grading need to run scripts/run_one_task.py (uv for uv.lock repos,
# pyyaml, numpy, openai). linux/amd64 so the contest's x86-64 wheels install; on Apple
# Silicon it runs under emulation. Built by distill/collect_local.sh.
FROM --platform=linux/amd64 python:3.13-slim

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends git patch protobuf-compiler pigz \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir pytest pytest-timeout==2.1.0 typer pdm-backend setuptools wheel poetry-core hatchling flit-core editables \
    && pip install --no-cache-dir pyyaml numpy openai uv

RUN git config --global user.email "agent@eval" \
    && git config --global user.name "Agent" \
    && git config --global init.defaultBranch main \
    && git config --global --add safe.directory '*' \
    && echo 'if [ -x /venv/bin/python ]; then export VIRTUAL_ENV=/venv PATH="/venv/bin:$PATH"; fi' > /etc/profile.d/venv.sh

# Tool commands run in login shells (bash -lc), and Debian's /etc/profile resets PATH, so the
# task venv is added in profile.d. The contest installs into the system Python instead.

ENV TEST_TMPDIR=/tmp
WORKDIR /workspace
