"""Join polynomial motions without resampling."""
import numpy as np
from scipy.interpolate import PPoly

def concatenate(first: PPoly, second: PPoly) -> PPoly:
    if first.c.shape[2:] != second.c.shape[2:]:
        raise ValueError("PPoly pieces must have matching output shape")
    degree_rows = max(first.c.shape[0], second.c.shape[0])

    def promote(coefficients: np.ndarray) -> np.ndarray:
        if coefficients.shape[0] == degree_rows:
            return coefficients
        padding = np.zeros(
            (degree_rows - coefficients.shape[0], *coefficients.shape[1:]), dtype=coefficients.dtype
        )
        return np.concatenate((padding, coefficients), axis=0)

    shift = float(first.x[-1])
    coefficients = np.concatenate((promote(first.c), promote(second.c)), axis=1)
    knots = np.r_[first.x, shift + second.x[1:]]
    return PPoly(coefficients, knots)
