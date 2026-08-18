# ArtraCFD Verification Tests

This directory contains verification and regression tests for the ArtraCFD solver.

## Tests

- `verify_gpu.c` — Runs one timestep in both CPU and GPU mode, compares node states
  to confirm bit-exact (or tolerance-bounded) agreement.

## Running Tests

```bash
cd test
make -f ../Makefile BUILD=gpu test_verify
```

Or from the project root:
```bash
make test_verify
