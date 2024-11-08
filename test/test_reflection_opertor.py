import numpy as np
from diffpy.structure import Lattice

from symmetry_breaking_measure import ReflectionOperator


def test_apply_reflection_against_ground_truth() -> None:
    # GIVEN
    original_xyz = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    axis = np.array([1, 0, 1])
    origin = np.array([0.5, 0.5, 0.5])
    lattice = Lattice(a=1, b=1, c=1, alpha=90, beta=90, gamma=90)
    expected_xyz = np.array(
        [
            [1, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 1],
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ]
    )

    # WHEN
    reflection_op = ReflectionOperator(axis=axis, origin=origin, lattice=lattice)
    transformed_xyz = reflection_op.apply(original_xyz)

    # THEN
    assert np.allclose(transformed_xyz, expected_xyz)
