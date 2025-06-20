import numpy as np
import time

def running_array(size: int) -> float:
    start = time.time()
    array = np.ones(size)*2
    array = np.power(array, 2)
    end = time.time()
    return end - start

def running_for_loop(size: int) -> float:
    start = time.time()
    list_ = [2] * size
    for i in range(size):
        list_[i] = list_[i] ** 2
    end = time.time()
    return end - start

size = int(1e4)
print(f"Running array operation for size:", running_array(size))
print("Running for loop operation for size :", running_for_loop(size))
print(f"Vectorized speedup: {running_for_loop(size) / running_array(size):.2f}x")
