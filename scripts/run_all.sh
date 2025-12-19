#!/usr/bin/env bash
set -euo pipefail

# Run the benchmark suite inside the base images and build an HTML report.
# Usage: SCALE=1.0 THREAD_SETS="1 8" ./scripts/run_all.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULT_DIR="$ROOT_DIR/results"
LOG_DIR="$RESULT_DIR/logs"
DATA_DIR="/workspace/.bench_data"
SCALE="${SCALE:-1.0}"
THREAD_SETS="${THREAD_SETS:-1 8}"

mkdir -p "$RESULT_DIR"
mkdir -p "$LOG_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker CLI not found. Please run this script on a host with Docker available." >&2
  exit 1
fi

echo "Running benchmarks at scale=$SCALE thread_sets=[$THREAD_SETS]"

run_anaconda() {
  local threads="$1"; local label="anaconda-${threads}t"; local image="continuumio/anaconda3:latest"
  echo "--> ${label} (image=${image}, threads=${threads})"
  docker run --rm -u $(id -u):$(id -g) -v "$ROOT_DIR:/workspace" -w /workspace "$image" bash -lc "set -eo pipefail; export HOME=/workspace; mkdir -p /workspace/.local; mkdir -p ${DATA_DIR}; PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin:/workspace/.local/bin; export PATH; PREFIX=/workspace/.local; rm -rf /workspace/.local/lib/python3.12/site-packages/numpy* /workspace/.local/lib/python3.12/site-packages/bottleneck* || true; PURELIB=\$(python - <<'PY'
import sysconfig
paths = sysconfig.get_paths(vars={'base': '/workspace/.local', 'platbase': '/workspace/.local'})
print(paths['purelib'])
PY
); export PURELIB; export PYTHONNOUSERSITE=1; export PYTHONPATH=/workspace:\${PURELIB}; python -m pip install --no-cache-dir --prefix /workspace/.local 'numpy<2' polars pyaml; python benchmarks/run_suite.py --label ${label} --output results/${label}.json --data-dir ${DATA_DIR} --scale ${SCALE} --threads ${threads}" 2>&1 | tee -a "$LOG_DIR/${label}.log"
}

run_intel() {
  local threads="$1"; local label="intelpython-${threads}t"; local image="intel/python:latest"
  echo "--> ${label} (image=${image}, threads=${threads})"
  docker run --rm -u $(id -u):$(id -g) -v "$ROOT_DIR:/workspace" -w /workspace "$image" bash -lc "set -eo pipefail; export HOME=/workspace; mkdir -p /workspace/.local; mkdir -p ${DATA_DIR}; export PATH=/opt/conda/bin:\$PATH; rm -rf /workspace/.local/lib/python3.12/site-packages/numpy* /workspace/.local/lib/python3.12/site-packages/bottleneck* || true; PURELIB=\$(python - <<'PY'
import sysconfig
paths = sysconfig.get_paths(vars={'base': '/workspace/.local', 'platbase': '/workspace/.local'})
print(paths['purelib'])
PY
); export PURELIB; export PYTHONNOUSERSITE=1; export PYTHONPATH=/workspace:\${PURELIB}; python -m pip install --no-cache-dir --prefix /workspace/.local 'numpy<2' polars pyaml; python benchmarks/run_suite.py --label ${label} --output results/${label}.json --data-dir ${DATA_DIR} --scale ${SCALE} --threads ${threads}" 2>&1 | tee -a "$LOG_DIR/${label}.log"
}

run_python_314() {
  local threads="$1"; local label="python-latest-slim-${threads}t"; local image="python:latest-slim"
  echo "--> ${label} (image=${image}, threads=${threads})"
  docker run --rm -u $(id -u):$(id -g) -v "$ROOT_DIR:/workspace" -w /workspace "$image" bash -lc "set -eo pipefail; export HOME=/workspace; mkdir -p /workspace/.local; PATH=/usr/local/bin:/usr/bin:/bin:/workspace/.local/bin; export PATH; USER_SITE=\$(python -c 'import site; print(site.getusersitepackages())'); export USER_SITE; PYTHONPATH=/workspace:\${USER_SITE}:\${PYTHONPATH:-}; export PYTHONPATH; python -m pip install -U pip --user && python -m pip install --no-cache-dir --user -r requirements.txt && python -m pip install --no-cache-dir --user tqdm pyaml; python benchmarks/run_suite.py --label ${label} --output results/${label}.json --data-dir ${DATA_DIR} --scale ${SCALE} --threads ${threads}" 2>&1 | tee -a "$LOG_DIR/${label}.log"
}

for THREADS in $THREAD_SETS; do
  echo "== Thread setting: ${THREADS} =="

  run_python_314 "${THREADS}"
  run_anaconda "${THREADS}"
  run_intel "${THREADS}"
done

echo "Building HTML report"
docker run --rm -u $(id -u):$(id -g) -v "$ROOT_DIR:/workspace" -w /workspace python:3.14-slim \
  bash -lc "export HOME=/workspace; python -m pip install -U pip --user && python -m pip install --no-cache-dir --user pandas jinja2 && python benchmarks/report.py --results results/*.json --output results/report.html"

echo "Done. See results/report.html"
