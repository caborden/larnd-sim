#!/usr/bin/env bash

set -u
set -o pipefail

result_dir="${1:-test-results/light-sim-triggers}"
mkdir -p "${result_dir}"

environment_file="${result_dir}/environment.txt"
log_file="${result_dir}/pytest.log"
junit_file="${result_dir}/pytest.junit.xml"

{
    date --iso-8601=seconds
    hostname
    git rev-parse HEAD
    git branch --show-current
    git status --short
    python --version
    python -c "import cupy; print('cupy', cupy.__version__)"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
} > "${environment_file}" 2>&1

set +e
python -m pytest \
    -vv \
    tests/testLightSimTriggers.py \
    --junitxml="${junit_file}" \
    2>&1 | tee "${log_file}"
test_status=${PIPESTATUS[0]}
set -e

echo "pytest_exit_status=${test_status}" | tee -a "${log_file}"
exit "${test_status}"
