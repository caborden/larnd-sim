import numpy as np
from numba.cuda.random import create_xoroshiro128p_states

from larndsim import detsim
from larndsim.consts import detector


def test_packed_current_preserves_waveforms_and_rng(monkeypatch):
    # Two tracks per launch, a partial final subbatch, an empty row and padded ticks.
    pixels = np.array([[-1, 1, -1, 0], [-1, -1, 1, -1],
                       [0, -1, 1, -1], [-1, -1, -1, -1], [-1, 0, -1, -1]], np.int32)
    radius = np.arange(pixels.size, dtype=np.float32).reshape(pixels.shape)
    packed = np.full((len(pixels), 2), -1, np.int32)
    packed_radius = np.full(packed.shape, -1, np.float32)
    slots = np.full_like(packed, -1)
    detsim.pack_neighbors[2, 4](pixels, radius, packed, packed_radius, slots)
    np.testing.assert_array_equal(slots, [[1, 3], [2, -1], [0, 2], [-1, -1], [1, -1]])
    valid = slots >= 0
    rows, cols = np.nonzero(valid)
    np.testing.assert_array_equal(packed[valid], pixels[rows, slots[valid]])
    np.testing.assert_array_equal(packed_radius[valid], radius[rows, slots[valid]])
    assert np.all(packed[~valid] == -1)

    settings = dict(N_PIXELS=(4, 4), PIXEL_PITCH=1., V_DRIFT=1., MIN_STEP_SIZE=.1,
                    TIME_SAMPLING=.1, RESPONSE_SAMPLING=.1, RESPONSE_BIN_SIZE=.25,
                    RESPONSE_MAX_TIME=2., TPC_BORDERS=np.array([[[0., 4.], [0., 4.], [0., 1.]]]))
    for key, value in settings.items():
        monkeypatch.setattr(detector, key, value)
    fields = 'x_start y_start z_start x_end y_end z_end long_diff tran_diff n_electrons'
    tracks = np.zeros(len(pixels), [(k, 'f8') for k in fields.split()] + [('pixel_plane', 'i4')])
    tracks['x_start'], tracks['x_end'] = .4, .7
    tracks['y_start'], tracks['y_end'] = .5, .5
    tracks['z_start'], tracks['z_end'] = .7, .9
    tracks['long_diff'], tracks['tran_diff'], tracks['n_electrons'] = .05, .1, 100.
    response = np.arange(8*8*21, dtype=np.float32).reshape(8, 8, 21)

    def run(neighbors, original_slots):
        states = create_xoroshiro128p_states(2 * pixels.shape[1] * 4, seed=42)
        initial = states.copy_to_host()
        signals = np.zeros((*neighbors.shape, 3), np.float32)
        for start in range(0, len(pixels), 2):
            batch = slice(start, start + 2)
            args = () if original_slots is None else (original_slots[batch], pixels.shape[1])
            detsim.tracks_current_mc[(2, neighbors.shape[1], 1), (1, 1, 4)](
                signals[batch], neighbors[batch], tracks[batch], response, states, *args)
        final = states.copy_to_host()
        assert np.any(final != initial)
        return signals, final

    dense_signals, dense_rng = run(pixels, None)
    packed_signals, packed_rng = run(packed, slots)
    assert np.any(dense_signals)
    restored = np.zeros_like(dense_signals)
    restored[rows, slots[valid]] = packed_signals[rows, cols]
    np.testing.assert_array_equal(restored.view(np.uint32), dense_signals.view(np.uint32))
    np.testing.assert_array_equal(packed_rng, dense_rng)
