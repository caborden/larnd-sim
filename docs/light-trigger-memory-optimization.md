# Light-trigger memory-transfer optimization

## Scope

This change updates `get_triggers` in `larndsim/light_sim.py`. Its public
signature and return tuple remain unchanged.

## Changes

- Beam-trigger mode now returns before signal grouping, resampling, and
  threshold comparison. Those calculations were previously performed even
  though beam triggers do not depend on the signal.
- Padding now uses `(-n_ticks) % sample_factor`. The previous expression added
  a full sampling interval when the signal length was already divisible by the
  sample factor, causing an unnecessary allocation and copy.
- Threshold state is retained per trigger group instead of being broadcast
  across every channel. It is expanded only over time to preserve the original
  light-tick timing behavior.
- The active optical-channel array is copied from device to host once and
  reused for all module metadata lookups. Previously `.get()` was called once
  before the module loop and again for every module.
- NumPy reduction of a CuPy signal array was replaced with an explicit CuPy
  reduction.
- All candidate trigger ticks are found on-device and transferred once per
  module. A host-side greedy pass applies the digitization dead time, replacing
  the former Python loop that synchronized the GPU for every accepted trigger
  and repeatedly sorted all remaining candidate ticks.
- Trigger output arrays are constructed and uploaded in bulk rather than from
  Python lists containing individual device scalars.
- Dead-time selection now compares absolute candidate ticks against one
  absolute `next_allowed_tick`. The former code sliced an already-shortened
  array using an absolute tick after the second trigger, which could suppress
  more than the configured digitization window.

## Expected effect

The threshold-trigger path trades many small device synchronizations and
scalar transfers for one channel-metadata transfer plus one candidate-index
transfer per active module. It also reduces temporary-array size during the
threshold calculation. The beam-trigger path avoids threshold work entirely.

The candidate-index transfer remains an intermediate design. If profiling
shows that transferring dense threshold candidates is costly, the next step
is a CUDA kernel that performs dead-time selection and output compaction on
the device.
