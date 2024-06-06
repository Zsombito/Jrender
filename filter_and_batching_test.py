from jrenderer.pipeline import Render
import jax
import jax.numpy as jnp
import time

xs = []
dim1 = []
dim2 = []
dim3 = []
dim4 = []
dim5 = []

for i in range(1, 8):
    array = jnp.ones(pow(10, i), float)
    mask = jax.random.randint(jax.random.key(0), [pow(10, i)], 0, 1, int).astype(bool)
    start = time.time_ns()
    jax.block_until_ready(array[mask])
    end = time.time_ns()
    xs.append(pow(10, i))
    dim1.append((end - start / 1000 / 1000))

    array = jnp.ones([pow(10, i), 3], float)
    mask = jax.random.randint(jax.random.key(0), [pow(10, i)], 0, 1, int).astype(bool)
    start = time.time_ns()
    jax.block_until_ready(array[mask, :])
    end = time.time_ns()
    dim2.append((end - start / 1000 / 1000))

    array = jnp.ones([pow(10, i), 300], float)
    mask = jax.random.randint(jax.random.key(0), [pow(10, i)], 0, 1, int).astype(bool)
    start = time.time_ns()
    jax.block_until_ready(array[mask, :])
    end = time.time_ns()
    dim3.append((end - start / 1000 / 1000))

    array = jnp.ones([pow(10, i), 3, 3], float)
    mask = jax.random.randint(jax.random.key(0), [pow(10, i)], 0, 1, int).astype(bool)
    start = time.time_ns()
    jax.block_until_ready(array[mask, :, :])
    end = time.time_ns()
    dim4.append((end - start / 1000 / 1000))


import matplotlib.pyplot as plt

plt.plot(xs, dim1, label="1D array")
plt.plot(xs, dim2, label="2D array (variable x 3)")
plt.plot(xs, dim3, label="2D array (variable x 300)")
plt.plot(xs, dim4, label="3D array (variable x 3 x 3)")
plt.legend()
plt.ylabel("Time duration of filtering (ms)")
plt.xlabel("Elements in the array")
plt.xscale("log")
plt.suptitle("Filtering arrays")
plt.savefig("./tests/FilteringTest.png")
plt.savefig("./tests/FilteringTest.svg")



batch_sizes = [6, 51, 501, 5001, 50001]

xs = []
times = []
for j in range(5):
    tmp = []

    for i in range(1, 8):
        array = jax.random.randint(jax.random.key(0), [pow(10, i), 3], 0, 1000, int)
        start = time.time_ns()
        batch, _ = jax.block_until_ready(Render.arrayBatcher(batch_sizes[j], array, [3]))
        end = time.time_ns()
        if j ==0:
            xs.append(pow(10, i))
        tmp.append((end - start) / 1000 / 1000)
    times.append(tmp)

#xs = []
#times = []
#for j in range(5):
    #tmp = []

    #for i in range(1, 8):
        #array = jax.random.randint(jax.random.key(0), [pow(10, i), 3], 0, 1000, int)
        #start = time.time_ns()
        #batch, _ = jax.block_until_ready(Render.arrayBatcher(batch_sizes[j], array, [3]))
        #end = time.time_ns()
        #if j ==0:
            #xs.append(pow(10, i))
        #tmp.append((end - start) / 1000 / 1000)
    #times.append(tmp)

plt.clf()
for i in range(5):
    plt.plot(xs, times[i], label=f"Batch size = {batch_sizes[i]}")


plt.legend()
plt.ylabel("Time duration of batching the array (ms)")
plt.xlabel("Elements in the array")
plt.xscale("log")
plt.suptitle("Batching arrays")
plt.savefig("./tests/ArrayBatchingTest.png")
plt.savefig("./tests/ArrayBatchingTest.svg")

        
    
    
