
import brax
import jax
import jax.numpy as jnp
from brax.envs import humanoid
from jrenderer.brax_adaptor import BraxRenderer




#Creating Brax Environment
human = humanoid.Humanoid()


#Initializing and configuring the renderer
def create_batch_envs(sys : brax.System, idx : int):
    renderer =  BraxRenderer.create(sys)
    return renderer.config({"X":160, "Y":90 })

def render_env(renderer : BraxRenderer, state : brax.State, loop_unroll):
    return renderer.renderState(state, loop_unroll)

def _render_batch_unjitted(renderers, state, loop_unroll = 10):
    return jax.vmap(render_env, [0, None, None])(renderers, state, loop_unroll)

render_batch = jax.jit(_render_batch_unjitted, static_argnames=["loop_unroll"])

Ids = jax.lax.iota(int, 30)
batched_envs = jax.vmap(create_batch_envs, [None, 0])(human.sys, Ids)

#Loading states
import pickle
with open('states.pkl', 'rb') as f:
    states : list[brax.State] = pickle.load(f)

#Generate frames
images = jnp.empty((0, 30, 90, 160, 3), int)
for i in range(100):
    pixels = jax.block_until_ready(jnp.array([render_batch(batched_envs, states[i].pipeline_state, 50)]))
    images = jnp.append(images, pixels, 0)
    print(f"Frame {i}/100")

import matplotlib.pyplot as plt
import numpy 
import matplotlib.animation as animation

fig, axs = plt.subplots(nrows=6, ncols=5, sharex=True, sharey=True, figsize=(15, 18))

frames = []
for f in range(100):
  per_frame = []
  for i in range(30):
    ax = axs[i // 5][i % 5]

    im = ax.imshow(numpy.asarray(images[f][i]))
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