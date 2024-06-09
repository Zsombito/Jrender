

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


config = {"X" : 24, "Y" : 24}

def render_env(renderer : BraxRenderer, state : brax.State, loop_unroll):
    return renderer.renderState(state, loop_unroll)

def _render_batch_unjitted(renderers, state, loop_unroll = 10):
    return jax.vmap(render_env, [0, None, None])(renderers, state, loop_unroll)

render_batch = jax.jit(_render_batch_unjitted, static_argnames=["loop_unroll"])

Render.loadVertexShaders(stdVertexShader, stdVertexExtractor)
Render.loadFragmentShaders(stdFragmentShader, stdFragmentExtractor)

human = humanoid.Humanoid()
Is = jax.lax.iota(int, 30)

print(f"Batched envs created")

import pickle

with open('states.pkl', 'rb') as f:
    states : list[brax.State] = pickle.load(f)
#print(states[0].pipeline_state.x.pos[3])
#print(states[80].pipeline_state.x.pos[3])

configs = [
    {"X" : 16, "Y" : 16},
    {"X" : 64, "Y" : 64},
    {"X" : 256, "Y" : 256},
    {"X" : 1280, "Y" : 720},
]

times = []
for t in range(4):
    tmp = []
    def create_batch_envs(sys : brax.System, idx : int):
        renderer =  BraxRenderer.create(sys)
        return renderer.config(configs[t])
    print(f"Running test {t}")

    batched_envs = jax.vmap(create_batch_envs, [None, 0])(human.sys, Is)
    for i in range(100):
        start = time.time_ns()
        pixels = jax.block_until_ready(jnp.array([render_batch(batched_envs, states[i].pipeline_state, 50)]))
        end = time.time_ns()
        if i != 0:
            tmp.append((end - start) / 1000 / 1000)
    
    times.append(tmp)


import matplotlib.pyplot as plt

frames = range(99)
for i in range(4):
    x, y = configs[i]["X"], configs[i]["Y"]
    plt.plot(frames, times[i], label = f"Resolution: {x}x{y}")

plt.legend()
plt.ylabel("Time taken to render batch (ms)")
plt.xlabel("Frame idx")
plt.suptitle("Batch Rendering Different Resolutions")
plt.savefig("./tests/BatchTestRender.png")
plt.savefig("./tests/BatchTestRender.svg")

pixels = []
avarages = []
for i in range(4):
    x, y = configs[i]["X"], configs[i]["Y"]
    pixels.append(x*y)
    avarages.append(sum(times[i]) / 99)

plt.clf()
plt.plot(pixels, avarages)
plt.ylabel("Time taken to render batch (ms)")
plt.xlabel("Pixel Count")
plt.suptitle("Batch Rendering Different Resolutions Comparison")
plt.savefig("./tests/BatchTestRenderComp.png")
plt.savefig("./tests/BatchTestRenderComp.svg")

    
    
batch_sizes = [5, 10, 50, 100]
times = []
for t in range(4):
    tmp = []
    def create_batch_envs(sys : brax.System, idx : int):
        renderer =  BraxRenderer.create(sys)
        return renderer.config({"X": 64, "Y":64})
    print(f"Running test {t}")

    Is = jax.lax.iota(int, batch_sizes[t])
    batched_envs = jax.vmap(create_batch_envs, [None, 0])(human.sys, Is)
    print("Envs created")
    for i in range(100):
        start = time.time_ns()
        pixels = jax.block_until_ready(jnp.array([render_batch(batched_envs, states[i].pipeline_state, 50)]))
        end = time.time_ns()
        if i != 0:
            tmp.append((end - start) / 1000 / 1000)
    
    times.append(tmp)


plt.clf()
frames = range(99)
for i in range(4):
    plt.plot(frames, times[i], label = f"Batch size: {batch_sizes[i]}")

plt.legend()
plt.ylabel("Time taken to render batch (ms)")
plt.xlabel("Frame idx")
plt.suptitle("Batch Rendering Different Batch Sizes")
plt.savefig("./tests/BatchTestBatch.png")
plt.savefig("./tests/BatchTestBatch.svg")

avrages = []
for i in range(4):
    avrages.append(sum(times[i]) / 99)

plt.clf()
plt.plot(batch_sizes, avrages)
plt.ylabel("Time taken to render batch (ms)")
plt.xlabel("Batch size")
plt.suptitle("Batch Rendering Different Batch Sizes Comparison")
plt.savefig("./tests/BatchTestBatchComp.png")
plt.savefig("./tests/BatchTestBatchComp.svg")
        
#@title ### Plot
#fig, axs = plt.subplots(nrows=6, ncols=5, sharex=True, sharey=True, figsize=(15, 18))

#frames = []
#for f in range(100):
  #per_frame = []
  #for i in range(30):
    #ax = axs[i // 5][i % 5]

    #im = ax.imshow(numpy.asarray(images[i][f]))
    #per_frame.append(im)

  #frames.append(per_frame)

#print("frames:", len(frames))

#ani = animation.ArtistAnimation(
    #fig,
    #frames,
    #interval=1, # 1fps
    #blit=True,
    #repeat=True,
    #repeat_delay=0,
#)
#ani.save('./brax_output/animation.gif', writer='pillow', fps=15)