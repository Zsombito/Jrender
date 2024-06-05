
import brax
from jrenderer.pipeline import Render
from jax import numpy as jnp
import jax

import brax
from brax.envs import ant, humanoid
from jrenderer.pipeline_brax_without_clipping import Render
from jrenderer.brax_adaptor import BraxRenderer
from jrenderer.shader import stdVertexExtractor, stdVertexShader, stdFragmentExtractor, stdFragmentShader
import time



Render.loadVertexShaders(stdVertexShader, stdVertexExtractor)
Render.loadFragmentShaders(stdFragmentShader, stdFragmentExtractor)

human = humanoid.Humanoid()

brax_renderer = BraxRenderer.create(human.sys)

import pickle

with open('states.pkl', 'rb') as f:
    states : list[brax.State] = pickle.load(f)

times = []
comp_times = []
avarage = 0
loop_unrolls = [1, 5, 10, 50, 100, 500, 1000]

for lp in range(7):
    print(f"Test{lp+1}")
    tmp = []
    for i in range(100):
        start = time.time_ns()
        pixels= jax.block_until_ready(brax_renderer.renderState(states[i].pipeline_state, loop_unroll = loop_unrolls[lp]))
        end = time.time_ns()
        if i == 0:
            comp_times.append((end - start) / 1000 / 1000 / 1000)
        else:
            tmp.append((end - start) / 1000 / 1000)
    
    times.append(tmp)

frames = range(99)
import matplotlib.pyplot as plt

for i in range(7):
    plt.plot(frames, times[i], label=f"Loop_unroll = {loop_unrolls[i]}")

plt.legend()
plt.ylabel("Time render (ms)")
plt.xlabel("Index of frame")
plt.suptitle("Effects of Loop Unroll in Rendering Times")
plt.savefig("./tests/loop_unroll_without_clipping.svg")
plt.savefig("./tests/loop_unroll_without_clipping.png")

plt.clf()
plt.plot(loop_unrolls, comp_times)
plt.ylabel("Time to compile (s)")
plt.xlabel("Loop unroll")
plt.suptitle("Effects of Loop Unroll in Compilation Times")
plt.savefig("./tests/loop_unroll_compilation.svg")
plt.savefig("./tests/loop_unroll_compilation.png")



