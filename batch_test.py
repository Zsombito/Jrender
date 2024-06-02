from jrenderer.pipeline import Render
import jax
import jax.numpy as jnp
import time

xs = []
ys = []

for i in range(1, 10):
    array = jnp.ones(pow(10, i), float)
    mask = jax.random.randint(jax.random.key(0), [pow(10, i)], 0, 1, int).astype(bool)
    start = time.time_ns()
    jax.block_until_ready(array[mask])
    end = time.time_ns()
    xs.append(pow(10, i))
    ys.append((end - start / 1000 / 1000))


import matplotlib.pyplot as plt

plt.plot(xs, ys)
plt.ylabel("Time duration of filtering (ms)")
plt.xlabel("Elements in the array")
plt.xscale("log")
plt.suptitle("Filtering arrays")
plt.savefig("FilteringTest.png")
