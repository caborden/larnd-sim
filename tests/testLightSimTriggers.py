#!/usr/bin/env python

"""GPU regression tests for ``larndsim.light_sim.get_triggers``."""

import numpy as np
import pytest


cp = pytest.importorskip('cupy')

try:
    _GPU_COUNT = cp.cuda.runtime.getDeviceCount()
except cp.cuda.runtime.CUDARuntimeError:
    _GPU_COUNT = 0

if _GPU_COUNT == 0:
    pytest.skip('get_triggers tests require a CUDA GPU', allow_module_level=True)

from larndsim import light_sim
from larndsim.consts import detector, light


N_CHANNELS = 8
CHANNELS_PER_GROUP = 2
DEFAULT_THRESHOLD = -1.0


@pytest.fixture(autouse=True)
def synthetic_light_geometry(monkeypatch):
    """Install a small, deterministic two-module optical geometry."""

    monkeypatch.setattr(light, 'LIGHT_TRIG_MODE', 0)
    monkeypatch.setattr(light, 'OP_CHANNEL_PER_TRIG', CHANNELS_PER_GROUP)
    monkeypatch.setattr(light, 'LIGHT_DIGIT_SAMPLE_SPACING', 2.0)
    monkeypatch.setattr(light, 'LIGHT_TICK_SIZE', 1.0)
    monkeypatch.setattr(light, 'LIGHT_TRIG_WINDOW', (0.0, 3.0))
    monkeypatch.setattr(
        light,
        'OP_CHANNEL_TO_TPC',
        np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64),
    )
    monkeypatch.setattr(
        light,
        'TPC_TO_OP_CHANNEL',
        np.asarray([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int64),
    )
    monkeypatch.setattr(detector, 'MODULE_TO_TPCS', {1: [0, 1], 2: [2, 3]})
    monkeypatch.setattr(detector, 'TPC_TO_MODULE', {0: 1, 1: 1, 2: 2, 3: 2})


def _device_inputs(signal, thresholds=None, channels=None):
    """Convert a host signal and its metadata to the normal device inputs."""

    if thresholds is None:
        thresholds = np.full(
            signal.shape[0] // CHANNELS_PER_GROUP,
            DEFAULT_THRESHOLD,
            dtype=np.float32,
        )
    if channels is None:
        channels = np.arange(signal.shape[0], dtype=np.int64)
    return cp.asarray(signal), cp.asarray(thresholds), cp.asarray(channels)


def _read_result(result):
    """Copy the three trigger outputs to host for assertions."""

    return tuple(cp.asnumpy(value) for value in result)


def _set_group_signal(signal, group_idx, ticks, value=-2.0):
    """Set one channel so the requested trigger group's sum crosses threshold."""

    signal[group_idx * CHANNELS_PER_GROUP, ticks] = value


def _reference_threshold_triggers(signal, thresholds, channels):
    """Pure NumPy implementation of the intended threshold/dead-time behavior."""

    n_channels, n_ticks = signal.shape
    n_groups = n_channels // light.OP_CHANNEL_PER_TRIG
    sample_factor = round(
        light.LIGHT_DIGIT_SAMPLE_SPACING / light.LIGHT_TICK_SIZE
    )

    signal_sum = signal.reshape(
        n_groups, light.OP_CHANNEL_PER_TRIG, n_ticks
    ).sum(axis=1)
    padding = (-n_ticks) % sample_factor
    if padding:
        signal_sum = np.pad(signal_sum, ((0, 0), (0, padding)))

    sampled_signal = signal_sum.reshape(
        n_groups, -1, sample_factor
    ).mean(axis=-1)
    group_above_threshold = np.repeat(
        sampled_signal, sample_factor, axis=-1
    )[:, :n_ticks]
    group_above_threshold = group_above_threshold < thresholds[:, np.newaxis]

    grouped_channels = channels.reshape(n_groups, light.OP_CHANNEL_PER_TRIG)
    tpc_ids = np.unique(light.OP_CHANNEL_TO_TPC[channels])
    module_ids = np.unique(
        [detector.TPC_TO_MODULE[tpc_id] for tpc_id in tpc_ids]
    )
    digit_ticks = int(
        np.ceil(
            (light.LIGHT_TRIG_WINDOW[0] + light.LIGHT_TRIG_WINDOW[1])
            / light.LIGHT_TICK_SIZE
        )
    )

    trigger_ticks = []
    trigger_channels = []
    for module_id in module_ids:
        module_tpcs = detector.MODULE_TO_TPCS[module_id]
        module_channels = light.TPC_TO_OP_CHANNEL[module_tpcs].ravel()
        module_group_mask = np.any(
            np.isin(grouped_channels, module_channels), axis=1
        )
        candidates = np.flatnonzero(
            np.any(group_above_threshold[module_group_mask], axis=0)
        )

        next_allowed_tick = 0
        for candidate in candidates:
            candidate = int(candidate)
            if candidate < next_allowed_tick:
                continue
            trigger_ticks.append(candidate)
            trigger_channels.append(module_channels)
            next_allowed_tick = candidate + digit_ticks

    if not trigger_ticks:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, n_channels), dtype=channels.dtype),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.asarray(trigger_ticks, dtype=np.int64),
        np.stack(trigger_channels).astype(channels.dtype, copy=False),
        np.zeros((len(trigger_ticks),), dtype=np.int64),
    )


def test_beam_trigger_first_subbatch(monkeypatch):
    """Beam mode returns one trigger without requiring threshold-compatible input."""

    monkeypatch.setattr(light, 'LIGHT_TRIG_MODE', 1)
    signal = cp.zeros((3, 5), dtype=cp.float32)
    channels = cp.asarray([9, 4, 2], dtype=cp.int32)

    ticks, trigger_channels, trigger_types = _read_result(
        light_sim.get_triggers(signal, cp.empty((0,)), channels, 0)
    )

    np.testing.assert_array_equal(ticks, [0])
    np.testing.assert_array_equal(trigger_channels, [[9, 4, 2]])
    np.testing.assert_array_equal(trigger_types, [1])
    assert trigger_channels.dtype == np.int32


def test_beam_trigger_later_subbatch_is_empty(monkeypatch):
    """Only the first subbatch emits the event's beam trigger."""

    monkeypatch.setattr(light, 'LIGHT_TRIG_MODE', 1)
    signal = cp.zeros((3, 5), dtype=cp.float32)
    channels = cp.asarray([9, 4, 2], dtype=cp.int32)

    ticks, trigger_channels, trigger_types = _read_result(
        light_sim.get_triggers(signal, cp.empty((0,)), channels, 1)
    )

    assert ticks.shape == (0,)
    assert trigger_channels.shape == (0, 3)
    assert trigger_types.shape == (0,)
    assert trigger_channels.dtype == np.int32


def test_threshold_mode_no_crossing_has_compatible_empty_shapes():
    """No crossing retains the established empty result contract."""

    signal = np.zeros((N_CHANNELS, 8), dtype=np.float32)
    device_inputs = _device_inputs(signal)
    ticks, trigger_channels, trigger_types = _read_result(
        light_sim.get_triggers(*device_inputs, 0)
    )

    assert ticks.shape == (0,)
    assert trigger_channels.shape == (0, N_CHANNELS)
    assert trigger_types.shape == (0,)


def test_threshold_mode_single_crossing():
    """A sampled threshold crossing produces its first light-tick index."""

    signal = np.zeros((N_CHANNELS, 8), dtype=np.float32)
    _set_group_signal(signal, group_idx=0, ticks=slice(2, 4))

    ticks, trigger_channels, trigger_types = _read_result(
        light_sim.get_triggers(*_device_inputs(signal), 0)
    )

    np.testing.assert_array_equal(ticks, [2])
    np.testing.assert_array_equal(trigger_channels, [[0, 1, 2, 3]])
    np.testing.assert_array_equal(trigger_types, [0])


def test_dead_time_uses_absolute_ticks_and_accepts_boundary():
    """Continuous crossings retrigger exactly when each dead-time window ends."""

    signal = np.zeros((N_CHANNELS, 12), dtype=np.float32)
    _set_group_signal(signal, group_idx=0, ticks=slice(2, 12))

    ticks, _, _ = _read_result(
        light_sim.get_triggers(*_device_inputs(signal), 0)
    )

    # digit_ticks is three: ticks 3/4, 6/7, and 9/10 are suppressed, while
    # ticks exactly at the next allowed boundary are accepted.
    np.testing.assert_array_equal(ticks, [2, 5, 8, 11])


def test_modules_apply_dead_time_independently():
    """A simultaneous crossing in two modules yields one trigger per module."""

    signal = np.zeros((N_CHANNELS, 8), dtype=np.float32)
    _set_group_signal(signal, group_idx=0, ticks=slice(2, 4))
    _set_group_signal(signal, group_idx=2, ticks=slice(2, 4))

    ticks, trigger_channels, trigger_types = _read_result(
        light_sim.get_triggers(*_device_inputs(signal), 0)
    )

    np.testing.assert_array_equal(ticks, [2, 2])
    np.testing.assert_array_equal(
        trigger_channels,
        [[0, 1, 2, 3], [4, 5, 6, 7]],
    )
    np.testing.assert_array_equal(trigger_types, [0, 0])


@pytest.mark.parametrize('n_ticks', [7, 8, 17])
@pytest.mark.parametrize('seed', [7, 19, 41])
def test_random_signals_match_numpy_reference(n_ticks, seed):
    """Divisible and padded signals agree with an independent CPU reference."""

    rng = np.random.default_rng(seed)
    signal = rng.uniform(-2.5, 0.5, size=(N_CHANNELS, n_ticks)).astype(
        np.float32
    )
    thresholds = rng.uniform(-2.0, -0.5, size=N_CHANNELS // 2).astype(
        np.float32
    )
    channels = np.arange(N_CHANNELS, dtype=np.int64)

    expected = _reference_threshold_triggers(signal, thresholds, channels)
    actual = _read_result(
        light_sim.get_triggers(*_device_inputs(signal, thresholds, channels), 0)
    )

    for actual_array, expected_array in zip(actual, expected):
        np.testing.assert_array_equal(actual_array, expected_array)


def test_invalid_trigger_group_size_is_rejected():
    """An incomplete trigger group fails with a useful error."""

    signal = cp.zeros((3, 8), dtype=cp.float32)
    thresholds = cp.zeros((1,), dtype=cp.float32)
    channels = cp.arange(3, dtype=cp.int64)

    with pytest.raises(ValueError, match='divisible'):
        light_sim.get_triggers(signal, thresholds, channels, 0)
