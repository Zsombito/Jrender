
import brax
from jrenderer.pipeline import Render
from jax import numpy as jnp

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
print(states[0].pipeline_state.x.pos[3])
print(states[80].pipeline_state.x.pos[3])

frames = []

for i in range(200):
    start = time.time_ns()
    pixels = jnp.transpose(brax_renderer.renderState(states[i].pipeline_state), [1, 0, 2])
    end = time.time_ns()
    print(f"Frame{i} took: {(end - start) / 1000 / 1000} ms")
    pixels = pixels.astype("uint8")
    frames.append(pixels)

print("Making giff")
import imageio
imageio.mimsave('./brax_output/output.gif', frames)