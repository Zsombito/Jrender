
import brax
from jrenderer.pipeline import Render
from jax import numpy as jnp
import jax

import brax
from brax.envs import ant, humanoid
from jrenderer.pipeline_brax_raster_optimized import Render
from jrenderer.brax_adaptor import BraxRenderer
from jrenderer.shader import stdVertexExtractor, stdVertexShader, stdFragmentExtractor, stdFragmentShader
import time



Render.loadVertexShaders(stdVertexShader, stdVertexExtractor)
Render.loadFragmentShaders(stdFragmentShader, stdFragmentExtractor)

human = humanoid.Humanoid()

brax_renderer = BraxRenderer.create(human.sys)
brax_renderer = brax_renderer.changeCameraLinker(1, 1)

import pickle

with open('states.pkl', 'rb') as f:
    states : list[brax.State] = pickle.load(f)

frames = []
avarage = 0

for i in range(100):
    if i == 5:
        with jax.profiler.trace("jax-trace-brax"):
            pixels= jax.block_until_ready(brax_renderer.renderState(states[i].pipeline_state))
    else:
        start = time.time_ns()
        pixels= jax.block_until_ready(brax_renderer.renderState(states[i].pipeline_state))
        end = time.time_ns()
    print(f"Frame{i} took: {(end - start) / 1000 / 1000} ms")
    frames.append(pixels)
    if i != 0:
        avarage += end - start

print(f"On avarage frames took: {avarage / 100 / 1000 / 1000} ms")

print("Making giff")
import imageio
imageio.mimsave('./brax_output/output.gif', frames)