# reference: https://github.com/cilinyan/VISA/blob/main/utils/random_list.py

import itertools

def lcg(modulus, a, c, seed):
    """Linear Congruential Generator"""
    while True:
        seed = (a * seed + c) % modulus
        yield seed

def get_random_number(probabilities, values, generator):
    assert len(probabilities) == len(values), "Length of probabilities and values must be the same"
    assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"

    random_float = next(generator) / 256.0  # Convert to a float between 0 and 1

    for value, accumulated_probability in zip(values, itertools.accumulate(probabilities)):
        if random_float < accumulated_probability:
            return value
    return values[-1]  # If due to floating-point precision no value was returned, return the last value

def get_random_list(probabilities, values, length, seed: int = 0):
    # Create a generator
    generator = lcg(modulus=256, a=1103515245, c=12345, seed=seed)
    return [get_random_number(probabilities, values, generator) for _ in range(length)]
