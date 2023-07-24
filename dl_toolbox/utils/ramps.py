import numpy as np
from functools import partial


def lin_ramp(current, start, end, start_val, end_val):
    if current < start:
        return start_val
    elif current < end:
        return start_val + ((current - start) / (end - start)) * (end_val - start_val)
    else:
        return end_val


def sigm_ramp(current, start, end, start_val, end_val):
    if current < start:
        return start_val
    elif current < end:
        return start_val + np.exp(-5 * (1 - (current - start) / (end - start)) ** 2) * (
            end_val - start_val
        )
    else:
        return end_val


class LinearRamp:
    def __init__(self, start, end, start_val, end_val):
        self.fn = partial(
            lin_ramp, start=start, end=end, start_val=start_val, end_val=end_val
        )

    def __call__(self, current):
        return self.fn(current)


class SigmoidRamp:
    def __init__(self, start, end, start_val, end_val):
        self.fn = partial(
            sigm_ramp, start=start, end=end, start_val=start_val, end_val=end_val
        )

    def __call__(self, current):
        return self.fn(current)
