# Running the light-trigger tests on NERSC

Run these commands from a checkout of the optimization branch on Perlmutter.
Replace `<gpu-account>` with the appropriate GPU allocation.

## Interactive GPU allocation

```bash
salloc \
  --nodes 1 \
  --qos interactive \
  --time 01:00:00 \
  --constraint gpu \
  --gpus 1 \
  --account <gpu-account>
```

Activate the same CUDA/CuPy environment used for LarND-sim, change to the
repository root, and verify that the allocated GPU is visible:

```bash
srun --ntasks=1 --gpus=1 nvidia-smi
```

## Run and save the focused results

```bash
srun --ntasks=1 --gpus=1 \
  tests/run_light_sim_trigger_tests.sh
```

The runner exits with pytest's status and creates:

- `test-results/light-sim-triggers/pytest.log`: readable test output;
- `test-results/light-sim-triggers/pytest.junit.xml`: structured JUnit result;
- `test-results/light-sim-triggers/environment.txt`: commit, branch, Python,
  CuPy, and GPU information.

Share all three files when reporting a result. They can be bundled with:

```bash
tar -czf light-sim-trigger-results.tar.gz \
  test-results/light-sim-triggers
```

## Run the full regression suite

After the focused tests pass:

```bash
mkdir -p test-results/full-suite
srun --ntasks=1 --gpus=1 \
  python -m pytest -vv \
  --junitxml=test-results/full-suite/pytest.junit.xml \
  2>&1 | tee test-results/full-suite/pytest.log
```
