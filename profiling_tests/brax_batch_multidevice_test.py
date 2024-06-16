import brax
from jrenderer.pipeline import Render
from jax import numpy as jnp
import jax

import brax
from brax.envs import ant, humanoid
from jrenderer.pipeline_brax_without_clipping import Render
from jrenderer.brax_adaptor import BraxRenderer
from jrenderer.shader import stdVertexExtractor, stdVertexShader, stdFragmentExtractor, stdFragmentShader
from functools import partial
import time
from jax.sharding import PositionalSharding
from jax.experimental import mesh_utils


def render_env(renderer : BraxRenderer, state : brax.State, loop_unroll):
    return renderer.renderState(state, loop_unroll)

@jax.jit
def render_batch(state, renderers):
    return jax.vmap(render_env, [0, None, None])(renderers, state, 50)


Render.loadVertexShaders(stdVertexShader, stdVertexExtractor)
Render.loadFragmentShaders(stdFragmentShader, stdFragmentExtractor)

human = humanoid.Humanoid()

print(f"Batched envs created")

import pickle

with open('states.pkl', 'rb') as f:
    stat : list[brax.State] = pickle.load(f)

states = []
for i in range(100):
    states.append(stat[i].pipeline_state)
#print(states[0].pipeline_state.x.pos[3])
#print(states[80].pipeline_state.x.pos[3])


configs = [
    {"X" : 16, "Y" : 16},
    {"X" : 64, "Y" : 64},
    {"X" : 256, "Y" : 256},
    {"X" : 1280, "Y" : 720},
]
timesA = []
for t in range(4):
    tmp = []
    def create_batch_envs(sys : brax.System, idx : int):
        renderer =  BraxRenderer.create(sys)
        return renderer.config(configs[t])

    Is = jax.lax.iota(int, 36)
    batched_envs = jax.vmap(create_batch_envs, [None, 0])(human.sys, Is)
    for i in range(100):
        start = time.time_ns()
        pixels = jax.block_until_ready(jnp.array([render_batch(states[i], batched_envs)]))
        end = time.time_ns()
        if i != 0:
            tmp.append((end - start) / 1000 / 1000)
        print(f"Running testA {t}: {i}/99 ({(end - start) / 1000 / 1000} ms)")
    
    timesA.append(tmp)

timesB = []
sharding = PositionalSharding(jax.devices()[:2])
for t in range(4):
    tmp = []
    def create_batch_envs(sys : brax.System, idx : int):
        renderer =  BraxRenderer.create(sys)
        return renderer.config(configs[t])
    
    def vmap_create_envs(idxs):
        return jax.vmap(create_batch_envs, [None, 0])(human.sys, idxs)
    
    @jax.jit
    def render_batch_pmap(batched_envs, states):
        render_batch_here = partial(render_batch, states)
        return jax.pmap(render_batch_here)(batched_envs)

    Is = jax.lax.iota(int, 36)
    batched_envs = jax.vmap(create_batch_envs, [None, 0])(human.sys, Is)
    batched_envs = jax.device_put(batched_envs, sharding.replicate(0))
    

    Is = jnp.array([jax.lax.iota(int, 18), jax.lax.iota(int, 18)])
    batched_envs = jax.pmap(vmap_create_envs)(Is)
    for i in range(100):
        start = time.time_ns()
        pixels = jax.block_until_ready(jnp.array([render_batch_pmap(batched_envs, states[i])]))
        end = time.time_ns()
        if i != 0:
            tmp.append((end - start) / 1000 / 1000)
        print(f"Running testB {t}: {i}/99 ({(end - start) / 1000 / 1000} ms)")
    
    timesB.append(tmp)


timesC = []
sharding = PositionalSharding(jax.devices()[:3])
for t in range(4):
    tmp = []
    def create_batch_envs(sys : brax.System, idx : int):
        renderer =  BraxRenderer.create(sys)
        return renderer.config(configs[t])
    
    def vmap_create_envs(idxs):
        return jax.vmap(create_batch_envs, [None, 0])(human.sys, idxs)
    
    @jax.jit
    def render_batch_pmap(batched_envs, states):
        render_batch_here = partial(render_batch, states)
        return jax.pmap(render_batch_here)(batched_envs)

    Is = jax.lax.iota(int, 36)
    batched_envs = jax.vmap(create_batch_envs, [None, 0])(human.sys, Is)
    batched_envs = jax.device_put(batched_envs, sharding.replicate(0))
    

    Is = jnp.array([jax.lax.iota(int, 18), jax.lax.iota(int, 18)])
    batched_envs = jax.pmap(vmap_create_envs)(Is)
    for i in range(100):
        start = time.time_ns()
        pixels = jax.block_until_ready(jnp.array([render_batch_pmap(batched_envs, states[i])]))
        end = time.time_ns()
        if i != 0:
            tmp.append((end - start) / 1000 / 1000)
        print(f"Running testC {t}: {i}/99 ({(end - start) / 1000 / 1000} ms)")
    
    timesC.append(tmp)

timesD = []
sharding = PositionalSharding(jax.devices()[:4])
for t in range(4):
    tmp = []
    def create_batch_envs(sys : brax.System, idx : int):
        renderer =  BraxRenderer.create(sys)
        return renderer.config(configs[t])
    
    def vmap_create_envs(idxs):
        return jax.vmap(create_batch_envs, [None, 0])(human.sys, idxs)
    
    @jax.jit
    def render_batch_pmap(batched_envs, states):
        render_batch_here = partial(render_batch, states)
        return jax.pmap(render_batch_here)(batched_envs)

    Is = jax.lax.iota(int, 36)
    batched_envs = jax.vmap(create_batch_envs, [None, 0])(human.sys, Is)
    batched_envs = jax.device_put(batched_envs, sharding.replicate(0))
    

    Is = jnp.array([jax.lax.iota(int, 18), jax.lax.iota(int, 18)])
    batched_envs = jax.pmap(vmap_create_envs)(Is)
    for i in range(100):
        start = time.time_ns()
        pixels = jax.block_until_ready(jnp.array([render_batch_pmap(batched_envs, states[i])]))
        end = time.time_ns()
        if i != 0:
            tmp.append((end - start) / 1000 / 1000)
        print(f"Running testD {t}: {i}/99 ({(end - start) / 1000 / 1000} ms)")
    
    timesD.append(tmp)

import matplotlib.pyplot as plt

#frames = range(99)
#for i in range(4):
    #x, y = configs[i]["X"], configs[i]["Y"]
    #plt.plot(frames, times[i], label = f"Resolution: {x}x{y}")

#plt.legend()
#plt.ylabel("Time taken to render batch (ms)")
#plt.xlabel("Frame idx")
#plt.suptitle("Batch Rendering Different Resolutions")
#plt.savefig("./tests/BatchTestRenderMulti.png")
#plt.savefig("./tests/BatchTestRenderMulti.svg")

pixels = []
avaragesA = []
avaragesB = []
avaragesC = []
avaragesD = []
for i in range(4):
    x, y = configs[i]["X"], configs[i]["Y"]
    pixels.append(x*y)
    avaragesA.append(sum(timesA[i]) / 99)
    avaragesB.append(sum(timesB[i]) / 99)
    avaragesC.append(sum(timesC[i]) / 99)
    avaragesD.append(sum(timesD[i]) / 99)

plt.clf()
plt.plot(pixels, avaragesA, label="GPUs used = 1")
plt.plot(pixels, avaragesB, label="GPUs used = 2")
plt.plot(pixels, avaragesC, label="GPUs used = 3")
plt.plot(pixels, avaragesD, label="GPUs used = 4")
plt.legend()
plt.ylabel("Time taken to render batch (ms)")
plt.xlabel("Pixel Count")
plt.suptitle("Rendering Times Varying With GPUs Used")
plt.savefig("./tests/BatchTestMulti.png")
plt.savefig("./tests/BatchTestMulti.svg")

    