# GPU memory-transfer audit

This is a static audit of code that may reduce GPU utilization through
host/device transfers, device synchronization, or avoidable full-array copies.
The findings are candidates for profiling, not measured bottlenecks. Line
numbers refer to the source at the time of the audit.

## Navigation and cross-file priorities

- [`larndsim/light_sim.py`](#larndsims-light_simpy)
- [`larndsim/active_volume.py`](#larndsims-active_volumepy)
- [`larndsim/detsim.py`](#larndsims-detsimpy)
- [`larndsim/drifting.py`](#larndsims-driftingpy)
- [`larndsim/fee.py`](#larndsims-feepy)
- [`larndsim/lightLUT.py`](#larndsims-lightlutpy)
- [`larndsim/pixels_from_track.py`](#larndsims-pixels_from_trackpy)
- [`larndsim/quenching.py`](#larndsims-quenchingpy)
- [Recommended profiling order](#recommended-profiling-order)

The ranking below combines transfer frequency, synchronization frequency,
array size, and how often the path is entered. It is a static estimate and
should be validated with representative Nsight Systems traces.

| Rank | File | Potential | Most important optimization |
|---:|---|---|---|
| 1 | [`light_sim.py`](#larndsims-light_simpy) | Very high | Replace the Python/device-scalar trigger loop in `get_triggers` with on-device trigger discovery; also eliminate repeated `.get()` calls. |
| 2 | [`pixels_from_track.py`](#larndsims-pixels_from_trackpy) | High at its call sites | Keep the `max_pixels` reduction result on-device or replace it with a device reduction whose scalar is transferred once; avoid recomputing the pixel expansion globally and again per subbatch. |
| 3 | [`active_volume.py`](#larndsims-active_volumepy) | Medium--high | Cache/upload TPC borders once and update a single mask in-place instead of allocating unused masks and a new full-length mask for every TPC. |
| 4 | [`fee.py`](#larndsims-feepy) | Medium | Preserve the deliberate CPU/HDF5 boundary, but batch and overlap its required device-to-host output copies; remove redundant device copies and broadcast materialization in digitization setup. |
| 5 | [`detsim.py`](#larndsims-detsimpy) | Medium, primarily device-memory traffic | Redesign or tile the large `signals` intermediate so it need not be fully written and then reread before reduction. No repeated explicit host/device copy was found in the module itself. |
| 6 | [`lightLUT.py`](#larndsims-lightlutpy) | Low | Keep the LUT and small channel maps resident across events/modules. The current explicit LUT upload is necessary, not an avoidable transfer. |
| 7 | [`drifting.py`](#larndsims-driftingpy) | Low | No memory-transfer fix is currently indicated; keep the in-place kernel fused with adjacent per-track transforms if profiling shows launch overhead matters. |
| 8 | [`quenching.py`](#larndsims-quenchingpy) | Low | No memory-transfer fix is currently indicated; possible fusion with `drift` is a launch/global-memory optimization, not a host/device-copy fix. |

<a id="larndsims-light_simpy"></a>

## `larndsim/light_sim.py`

### Summary

The highest-priority region is `get_triggers` (lines 447--509). It explicitly
copies `op_channel_idx` from device to host more than once, mixes NumPy and
CuPy operations, synchronizes the device repeatedly while finding triggers,
and then rebuilds device arrays from Python lists. `sim_triggers` (lines
597--650) keeps most work on the GPU, but its Python branches synchronize on
device reductions and its padding strategy repeatedly reallocates and copies
large signal and truth arrays.

| Priority | Lines | Transfer/copy candidate | Why it matters | Candidate direction |
|---|---:|---|---|---|
| High | 447, 458 | `op_channel_idx.get()` is called twice | Each call is an explicit device-to-host copy and synchronization. The second transfer duplicates the first. `np.unique`/`np.isin` then perform CPU work before the mask is used to index a CuPy array. | At minimum, call `.get()` once and reuse it. Prefer keeping channel-to-module mapping and membership masks on-device, or precompute these small/static mappings outside the event loop. |
| High | 460 | `np.any(sample_above_thresh[op_channel_mask], axis=0)` mixes a host-created mask with a device array | The host Boolean mask must be made usable by the GPU indexing path, and the advanced-index operation materializes a device temporary. The `np.any` spelling obscures whether dispatch remains on device. | Use an explicitly device-resident mask and `cp.any`. Better, avoid the gathered temporary by reducing channels through a precomputed module/group mapping. |
| High | 463--473 | `while cp.any(...)` and device scalars in Python control flow | Every loop test must copy a scalar result to the host/synchronize. `next_idx` remains a device scalar, then is used for Python slicing/arithmetic, introducing further scalar synchronizations. `cp.sort(cp.nonzero(...)[0])[0]` also allocates and sorts all matches merely to obtain the first. | Replace the Python loop with a GPU trigger-finding kernel/vectorized edge and dead-time calculation. As an intermediate improvement, use `cp.argmax`/first-nonzero logic and make the synchronization explicit and once per trigger. |
| High | 509 | `cp.array(trigger_idx_list)` and `cp.array(trigger_type_list)` rebuild arrays from lists of device scalars | This can cause scalar device-to-host reads followed by a host-to-device array construction, in addition to Python overhead. | Accumulate indices entirely on-device, or, if trigger discovery remains CPU-controlled, explicitly transfer once, build NumPy outputs, and upload once only if downstream kernels require device arrays. |
| High | 608--625 | Device reductions feed Python `if`/`int`, followed by `cp.concatenate` padding | `trigger_idx.min()`/`padded_trigger_idx.max()` synchronize to obtain host scalars. Each concatenate allocates a new buffer and copies the whole signal or truth tensor; up to three large arrays are recopied at each end. | Obtain the two extrema in one deliberate synchronization. Allocate final padded arrays once, initialize them, and copy the original arrays into slices. Consider teaching `digitize_signal` to handle out-of-range samples so physical padding is unnecessary. |
| High | 631--644 | Missing-channel test and append/sort pipeline | The Python `if cp.any(...)` synchronizes. `cp.isin` is evaluated multiple times. Four concatenations and four `take_along_axis` calls allocate and copy the signal and potentially very large truth tensors. | Compute the membership mask once. Prefer a fixed channel layout or an indirect channel map so missing channels need not be appended and all tensors need not be sorted. |
| Medium | 628 | `cp.array(gen_light_detector_noise(...))` | `gen_light_detector_noise` already returns a CuPy array. `cp.array` defaults to a copy, so this creates an avoidable full-size device-to-device copy before addition. | Remove the wrapper: `signal += gen_light_detector_noise(...)`. |
| Medium | 509 | `cp.array(op_channel_idx_list)` where entries are a mixture of NumPy and CuPy arrays | Threshold mode stores host `op_channels`; beam mode stores device `op_channel_idx`. Constructing one CuPy output from these Python containers entails host-to-device copies and makes behavior mode-dependent. | Use one memory domain consistently and construct the final array once. Static trigger-channel layouts can often be cached on device. |
| Medium | 38--41 | CuPy reductions/scalars are consumed by Python in `get_nticks` | The caller passes `light_inc`, a CuPy slice. NumPy calls generally dispatch to CuPy here, but `if np.any(mask)`, `int(np.ceil(...))`, and returning scalar start/end times force synchronization/device-to-host scalar extraction. Boolean indexing also creates temporary device arrays. | Implement a small device reduction returning min/max/count, then perform one explicit scalar transfer. Profile whether the event-level reduction cost justifies a custom kernel. |
| Medium | 54--56 | `get_active_op_channel` mixes NumPy calls with a CuPy input, then wraps the result in `cp.array` | With a CuPy input, the reductions/`where` remain device-side through dispatch, but the Boolean test synchronizes and the final `cp.array` may copy an already-device-resident result. This function is currently commented out at its main call site. | Use `cp.any`, `cp.where`, and `.astype(cp.int32, copy=False)`; avoid the separate `if` because `cp.where` naturally returns an empty array. |
| Medium | 597--599 | Full-size output initialization, especially truth arrays | These are necessary outputs, but the two truth tensors can dominate bandwidth and memory footprint. Initializing them is a full device-memory write even when truth tracking is disabled or sparse. | Skip allocation/kernels when `MAX_MC_TRUTH_IDS == 0`; longer term, consider sparse/compacted truth storage. |
| Low | 382--404 | Noise generation creates several full-size temporaries | Spectrum interpolation, random phase, complex exponential, multiplication, inverse FFT, rounding, and optional concatenation each allocate device buffers. These are device-to-device rather than host/device copies, but can be memory-bandwidth and allocator heavy. | Profile CuPy memory-pool reuse; use `out=`/in-place operations where supported, and generate the exact FFT-compatible output shape without concatenation. |
| Low | 641--644 | `np.newaxis` appears in CuPy indexing expressions | `np.newaxis` is just `None` and does not itself transfer memory. It is not a problem, though explicit `cp` operations would make device intent clearer. | No change required for transfer avoidance. |

### Transfers caused by how the functions are called

These transfers are not written as `.get()` inside `light_sim.py`, but are
important to the behavior of its CUDA kernels:

- `cli/simulate_pixels.py` constructs `sorted_indices`, `scint_model`, and
  `sipm_response` as NumPy arrays and passes them to CUDA kernels
  `sum_light_signals`, `calc_scintillation_effect`, and
  `calc_light_detector_response`. Numba must make host data available to the
  device at launch. These arrays are recreated inside the event loop, so the
  host-to-device cost can recur for every batch/event. Construct them with
  CuPy, explicitly upload once, and cache invariant response arrays when their
  configuration/shape is unchanged.
- The `save_results` path in `cli/simulate_pixels.py` deliberately converts
  accumulated CuPy arrays to NumPy before calling the HDF5 export functions.
  The resulting device-to-host transfer is required by the current h5py-based
  output boundary. It should be treated separately from unnecessary transfers;
  optimization options include fewer/larger batched transfers, pinned host
  buffers, and overlap with computation.
- Consequently, the NumPy operations in `zero_suppress_waveform_truth`,
  `export_light_wvfm_to_hdf5`, and `export_light_trig_to_hdf5` operate on host
  arrays in the current main path. They are not additional GPU transfers at
  present, although their docstrings do not state this contract.

### Recommended profiling order

1. Instrument `get_triggers` with Nsight Systems and count CUDA API
   synchronizations and memcpy operations per event.
2. Remove the redundant `cp.array` at line 628 and cache the two
   `op_channel_idx.get()` results as a low-risk baseline change.
3. Compare the current padding/concatenate path with one-shot allocation or
   bounds-aware digitization using representative truth-array sizes.
4. Move/cache `sorted_indices`, `scint_model`, and `sipm_response` on-device,
   then measure kernel-launch transfer time separately from kernel execution.
5. Record transfer bytes, synchronization count, GPU active time, and total
   event time for each change; GPU utilization alone can hide improvements or
   regressions in end-to-end throughput.

<a id="larndsims-active_volumepy"></a>

## `larndsim/active_volume.py`

### Assessment

This function correctly selects NumPy or CuPy through
`cp.get_array_module(track_seg)`, so it does not inherently force track data to
the host. In the main simulation, however, `track_seg` is device-resident while
`detector.TPC_BORDERS`/`module_borders` originate as NumPy arrays. That makes
the generic conversion at line 24 and the repeated full-array mask expressions
the main concerns.

| Priority | Lines | Transfer/copy candidate | Why it matters | Candidate direction |
|---|---:|---|---|---|
| High | 24; caller lines 909, 966--967 | `xp.sort(tpc_borders, axis=-1)` with `xp is cp` and host `tpc_borders` | The call must make the host border array available on-device each time `select_active_volume` runs. In module mode it is called repeatedly. Sorting also creates a fresh border array even though detector borders are configuration data and do not change during a call. | Normalize/sort borders once when detector properties are loaded, cache a CuPy copy per configuration/module, and pass the device array explicitly. |
| Medium | 21--23 | Three full-size Boolean arrays are allocated, but `tpc_start_mask` and `tpc_end_mask` are never used | Two device allocations and initializations are entirely unnecessary. | Remove the two unused arrays. |
| Medium | 29--44 | `tpc_mask = tpc_mask | (...)` allocates a new full-length mask for every TPC | The comparison expression creates temporaries, and the non-in-place OR recopies the accumulated mask each iteration. Cost grows with segments times TPC count. | Use `tpc_mask |= ...`; preferably compute start/end containment with fused CuPy elementwise logic or a CUDA kernel that loops over TPCs per segment. |
| Low | 46 | `xp.nonzero(tpc_mask)[0]` creates an index array | This output is required by current callers, so it is not an unnecessary copy. Returning a Boolean mask might avoid an index-array allocation only if all downstream indexing is changed accordingly. | Keep unless profiling shows compaction/index construction is significant. |

The cached-border recommendation also benefits the constant-geometry access
patterns discussed in [`drifting.py`](#larndsims-driftingpy),
[`lightLUT.py`](#larndsims-lightlutpy), and
[`pixels_from_track.py`](#larndsims-pixels_from_trackpy).

<a id="larndsims-detsimpy"></a>

## `larndsim/detsim.py`

### Assessment

No explicit host/device copy or host-side CuPy scalar extraction occurs in
this module. Its kernels receive device arrays in the main path, including the
induction `response`, which `detector.load_response` loads with CuPy. The main
opportunity is device-memory traffic and intermediate size rather than PCIe
traffic.

| Priority | Lines | Transfer/copy candidate | Why it matters | Candidate direction |
|---|---:|---|---|---|
| High | 106--213, then 216--281 | `tracks_current_mc` writes the full `(segments, pixels, ticks)` `signals` tensor and `sum_pixel_signals` reads it back | This can be the largest charge-simulation intermediate. It consumes device memory, incurs a full global-memory write/read, and drives subbatching. It is not a host/device transfer, but it can limit utilization through memory pressure. | Profile a tiled/fused accumulation into per-pixel outputs, or process smaller tick/pixel tiles with on-device reduction. Preserve truth bookkeeping and atomic-contention behavior when comparing designs. |
| Medium | 188, 196--197 | RNG state is accessed three times with the same state index inside every integration step | This is repeated global device-memory traffic and may serialize state updates. It is not a host copy. | Load/update a per-thread RNG state more locally or generate the required normal values together if supported by the chosen Numba RNG implementation. Validate statistical behavior. |
| Medium | 284--357 | Pixel/track map kernels repeatedly scan `pixels` for every unique pixel | This rereads the same device arrays many times and writes a dense padded map. | Reuse the already available inverse mapping from the caller or build a sorted/CSR-style association once. This may also shrink the downstream truth tensors. |
| Low | 7 | `cupy` is imported but unused | No runtime copy results, but it can mislead readers about host-side array manipulation in this module. | Remove the unused import during cleanup. |

At the call site, `int(cp.max(unique_pix))` and `int(num_backtrack.sum())`
introduce scalar synchronizations while sizing arrays. Those are documented as
call-boundary findings rather than defects inside `detsim.py`; reducing dense
intermediate sizing would address both. See also the truth-array allocation
pressure in [`light_sim.py`](#larndsims-light_simpy).

<a id="larndsims-driftingpy"></a>

## `larndsim/drifting.py`

### Assessment

No potentially unnecessary host/device transfer was found. `drift` is a
single in-place CUDA kernel over the already-device-resident track array.
Detector constants are compiled kernel globals rather than copied through
Python on every track.

| Priority | Lines | Observation | Candidate direction |
|---|---:|---|---|
| Low | 30--60 | Each track is loaded into a local structured scalar and its fields are written back in-place. There is no host round trip. | No transfer-specific change. If profiling shows the short kernel is launch- or bandwidth-bound, consider fusing it with [`quenching.quench`](#larndsims-quenchingpy), while checking register pressure and maintainability. |
| Low | 34--39 | Every thread scans all TPC borders to determine its plane. | This is device constant/global-memory and control-flow cost, not host/device copying. A precomputed plane assignment or spatial lookup may help only if this kernel is measured as significant. |

<a id="larndsims-feepy"></a>

## `larndsim/fee.py`

### Assessment

`fee.py` has a GPU compute path (`gen_event_times`, `digitize`, and
`get_adc_values`) and deliberately CPU-bound packet/HDF5 export paths. The main
simulation converts accumulated results to NumPy before calling
`export_to_hdf5`, so the NumPy work in lines 138--377 does not itself trigger
additional GPU transfers. The unavoidable output boundary should not be
confused with avoidable copies.

| Priority | Lines | Transfer/copy candidate | Why it matters | Candidate direction |
|---|---:|---|---|---|
| High | 84--377; `save_results` caller | All charge outputs are transferred to host before packet construction/HDF5 writes | h5py and LArPix packet construction require host data in the current design, so a device-to-host transfer is necessary. Synchronous conversion of many separate arrays can leave the GPU idle. | Transfer fewer, larger batches into pinned host buffers and overlap export with the next compute batch where thread/process safety permits. Measure end-to-end memory growth and I/O backpressure. |
| Medium | 531--536 | `digitize` builds a chain of elementwise temporaries for multiply/add, clamp, scale, floor, and the zero mask | CuPy keeps these on-device, but multiple full-array reads/writes can be costly for the ADC tensor. | Use a fused CuPy elementwise kernel or CUDA kernel, including the zero-input rule. |
| Medium | caller lines 248--255 | `cp.array(pixel_*_lut[...])` and multiplication by `cp.ones` | If the CUDA dictionary lookup already returns a device array, `cp.array` copies it. Multiplying by a full ones array materializes gain/pedestal matrices even though broadcasting can supply the same values. | Use `cp.asarray`/the lookup result directly and pass `pixel_values[:, None]` so `digitize` broadcasts without a tiled copy. |
| Medium | 153--160, 176--300 | Export loops over every pixel/ADC in Python after arrays are on host | This is CPU/object-construction overhead rather than a GPU transfer, but it extends the period before buffers can be released and may reduce pipeline overlap. | Profile vectorized packet-field preparation or chunked producer/consumer export. |
| Low | 66--81 | Event times are generated on-device and eventually copied to host for output | They are also used in GPU simulation, so device generation is reasonable. The final transfer is small and required. | No isolated change; include it in a batched output transfer. |
| Low | 388--517 | Sync/timestamp exporters repeatedly create small NumPy arrays inside Python loops | Host-only allocation overhead; no GPU copy in the current path. | Preallocate structured association arrays if packet export becomes CPU-bound. |

The same output-boundary strategy applies to light waveform export; see
[`light_sim.py` transfers caused by how functions are called](#transfers-caused-by-how-the-functions-are-called).

<a id="larndsims-lightlutpy"></a>

## `larndsim/lightLUT.py`

### Assessment

No unnecessary transfer appears inside `calculate_light_incidence`. The main
caller loads the LUT on the CPU for validation/modification, uploads it once
with `to_device`, and passes device-resident channel efficiency/mapping arrays.
That upload is necessary under the current loading workflow.

| Priority | Lines | Observation | Candidate direction |
|---|---:|---|---|
| Medium | 66--135; caller lines 1029--1053 | The potentially very large LUT is uploaded when its file/configuration changes | This is a real host-to-device transfer, but the caller already avoids reloading an unchanged LUT across modules. | Retain/codify the cache keyed by LUT file, device, dtype, and relevant configuration. Consider loading directly into device-backed storage only if measured startup/module-switch time warrants it. |
| Medium | 108--134 | Each track reads a LUT voxel and loops over all output channels, writing a large structured output | Heavy device global-memory traffic, but no host copy. Access coalescing may be limited because one thread walks channels serially. | Profile a 2-D track/channel launch and structure-of-arrays outputs, especially when optical-channel count is large. Account for LUT reuse/cache behavior. |
| Low | 15--63 | `get_voxel` is `@nb.njit` and is called from a CUDA kernel | Numba compiles this helper for device use in this context; it does not send positions to the host. | No transfer-specific change. Marking it explicitly as a CUDA device function could clarify intent if compatible with CPU uses/tests. |

The resulting device arrays feed [`light_sim.py`](#larndsims-light_simpy),
where the more substantial synchronization and copy opportunities occur.

<a id="larndsims-pixels_from_trackpy"></a>

## `larndsim/pixels_from_track.py`

### Assessment

The kernels themselves contain no explicit host/device copy. The main call
pattern, however, passes a one-element NumPy array to `max_pixels` and then
immediately uses the result on the CPU to size CuPy outputs. This causes a
host-to-device copy before the kernel and device-to-host copy/synchronization
after it, both globally and for every selected subbatch.

| Priority | Lines | Transfer/copy candidate | Why it matters | Candidate direction |
|---|---:|---|---|---|
| High | 54--83; caller lines 1222, 1309 | `max_pixels` receives host `np.array([0])` outputs | Numba must stage the scalar accumulator on-device and copy its result back. The required result also forces synchronization before allocation sizes can be calculated. This repeats per batch/subbatch. | Allocate a persistent one-element device array, reset it on-device, launch the reduction, and perform one explicit scalar transfer only when the host truly must size arrays. Better, derive/cap allocation sizes on-device or use reusable capacity-based buffers. |
| High | 85--137; caller lines 1222--1327 | Active/neighboring pixels are computed for all selected tracks, then recomputed for subbatch-selected tracks | Repeated kernels rewrite large padded device arrays and repeat geometry work. It also repeats the `max_pixels` round trip above. | Reuse/gather the global pixel results for subbatches, or reorganize batching so pixel expansion occurs exactly once per track. Compare gather-copy cost against recomputation. |
| Medium | 103--117, 133--137 | Dense padded output arrays are allocated using worst-case pixel counts | Initialization and writes scale with `tracks * maximum`, even when most rows are short. | Use counts plus prefix sums and compact/CSR outputs, or maintain reusable capacity buffers. This would also simplify mappings in [`detsim.py`](#larndsims-detsimpy). |
| Medium | 238--298 | Duplicate suppression linearly scans the full `neighboring_pixels` row for every candidate | This repeatedly reads device memory and can dominate for large radii. It is not a host transfer. | Use a bounded local bitmap/tile mask, sorted generation, or a compact uniqueness scheme suited to the small geometric neighborhood. |
| Low | 12--52, 139--237 | Numba helpers execute as device functions when called by CUDA kernels | Despite NumPy-oriented docstrings, these calls do not transfer values to the CPU. | Clarify array/device contracts in docstrings; no transfer fix required. |

<a id="larndsims-quenchingpy"></a>

## `larndsim/quenching.py`

### Assessment

No potentially unnecessary host/device transfer was found. Like
[`drifting.py`](#larndsims-driftingpy), `quench` updates the device-resident
structured track array in-place.

| Priority | Lines | Observation | Candidate direction |
|---|---:|---|---|
| Low | 23--53 | Per-track fields are read and written entirely inside one CUDA kernel. | No transfer-specific change. Consider fusion with the immediately following `drift` kernel only after profiling launch/global-memory costs; fusion can increase register pressure and coupling. |
| Low | 47, 50 | Device exceptions exist on invalid configuration/data paths | This is correctness/control-flow behavior, not routine memory copying. | Validate `mode` on the host before launch and validate pathological inputs separately if exception handling appears in profiles. |

## Audit status

| Module | Status |
|---|---|
| `larndsim/light_sim.py` | Initial static audit complete |
| `larndsim/active_volume.py` | Initial static audit complete |
| `larndsim/detsim.py` | Initial static audit complete |
| `larndsim/drifting.py` | Initial static audit complete |
| `larndsim/fee.py` | Initial static audit complete |
| `larndsim/lightLUT.py` | Initial static audit complete |
| `larndsim/pixels_from_track.py` | Initial static audit complete |
| `larndsim/quenching.py` | Initial static audit complete |
| Other `larndsim` modules | Not yet audited |
