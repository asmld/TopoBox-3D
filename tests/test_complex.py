import numpy as np

from topobox3d.complex import build_complex


def test_single_tetrahedron_is_a_chain_complex():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    payload = build_complex(points, np.array([[0, 1, 2, 3]]), beta1=0, beta2=0)

    assert payload["edges"].shape == (6, 2)
    assert payload["faces"].shape == (4, 3)
    assert payload["oriented_tetra"].shape == (1, 4)
    np.testing.assert_array_equal(payload["chain_complex_error"], np.zeros(2))
    assert np.isclose(payload["tetra_volumes"].sum(), 1.0 / 6.0)
