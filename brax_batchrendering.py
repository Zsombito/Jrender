

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


def create_batch_envs(sys : brax.System, idx : int):
    return BraxRenderer.create(sys)

def render_env(renderer : BraxRenderer, state : brax.State):
    return renderer.renderState(state)

@jax.jit
def render_batch(renderers, state):
    return jax.vmap(render_env, [0, None])(renderers, state)

Render.loadVertexShaders(stdVertexShader, stdVertexExtractor)
Render.loadFragmentShaders(stdFragmentShader, stdFragmentExtractor)

human = humanoid.Humanoid()
Is = jax.lax.iota(int, 30)

batched_envs = jax.vmap(create_batch_envs, [None, 0])(human.sys, Is)

print(f"Batched envs created")

import pickle

with open('states.pkl', 'rb') as f:
    states : list[brax.State] = pickle.load(f)
#print(states[0].pipeline_state.x.pos[3])
#print(states[80].pipeline_state.x.pos[3])

avarage = 0
frames = jnp.empty((0, 30, 180, 260, 3), "uint8")

for i in range(100):
    start = time.time_ns()
    pixels = jax.block_until_ready(jnp.array([render_batch(batched_envs, states[i].pipeline_state)]))
    end = time.time_ns()
    frames = jnp.append(frames, pixels, 0)
    print(f"Frame{i} for batch took: {(end - start) / 1000 / 1000} ms")
    #print(f"On avarage frames took: {avarage / 100 / 1000 / 1000} ms")


    #if i == 5:
        #with jax.profiler.trace("jax-trace-brax"):
            #pixels= jax.block_until_ready(brax_renderer.renderState(states[i].pipeline_state))
    #else:
        #start = time.time_ns()
        #pixels= jax.block_until_ready(brax_renderer.renderState(states[i].pipeline_state))
        #end = time.time_ns()
    #frames.append(jnp.transpose(pixels, [1,0,2]).astype("uint8"))
    #if i != 0:
        #avarage += end - start

#print(f"On avarage frames took: {avarage / 100 / 1000 / 1000} ms")
images = jnp.transpose(frames, [1, 0, 2, 3, 4])

print("Making giff")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy

        
#@title ### Plot
fig, axs = plt.subplots(nrows=6, ncols=5, sharex=True, sharey=True, figsize=(15, 18))

frames = []
for f in range(100):
  per_frame = []
  for i in range(30):
    ax = axs[i // 5][i % 5]

    im = ax.imshow(numpy.asarray(images[i][f]))
    per_frame.append(im)

  frames.append(per_frame)

print("frames:", len(frames))

ani = animation.ArtistAnimation(
    fig,
    frames,
    interval=1, # 1fps
    blit=True,
    repeat=True,
    repeat_delay=0,
)
ani.save('./brax_output/animation.gif', writer='pillow', fps=15)