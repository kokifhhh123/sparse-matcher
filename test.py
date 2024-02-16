import numpy as np
import timeit

# Example arrays to append or concatenate
num_arrays = 10000
array_size = 3

arrays_to_append = [np.array([i, i + 1, i + 2]) for i in range(num_arrays)]

# Using np.append in a loop
def append_in_loop():
    result_array = np.array([])
    for arr in arrays_to_append:
        result_array = np.append(result_array, arr)
    return result_array.reshape((-1, array_size))

# Using np.vstack
def vstack_directly():
    return np.vstack(arrays_to_append)

# Measure execution time
# time_append = timeit.timeit(append_in_loop, number=1000)
time_vstack = timeit.timeit(vstack_directly, number=1000)

# print(f"Time for np.append in a loop: {time_append:.6f} seconds")
print(f"Time for np.vstack: {time_vstack:.6f} seconds")

