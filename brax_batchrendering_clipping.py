

import brax
from jax import numpy as jnp
import jax

import brax
from brax.envs import ant, humanoid
from jrenderer.pipeline_brax import Render
from jrenderer.brax_adaptor_with_cliping import BraxRenderer
from jrenderer.shader import stdVertexExtractor, stdVertexShader, stdFragmentExtractor, stdFragmentShader
from jaxtyping import Array
import time


def create_batch_envs(sys : brax.System, idx : int):
    return BraxRenderer.create(sys)

@jax.jit
def renderEnvsA(renders, state : brax.State):
    def map(render : BraxRenderer, state : brax.State):
        return render.render_partA(state)
    
    return jax.vmap(map, [0, None])(renders, state)

@jax.jit
def renderEnvsC(renders, batched_faces, faces, pos3s, norms, perVertices, shaded_perVertices):
    def map(render : BraxRenderer, batched_face, face, pos3, norm, perVertex, shaded_perVertex):
        return jnp.transpose(render.render_partC(batched_face, face, pos3, norm, perVertex, shaded_perVertex))
    
    return jax.vmap(map, [0, 0, 0, 0, 0, 0, 0])(renders, batched_faces, faces, pos3s, norms, perVertices, shaded_perVertices)

    


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
    batched_envs, ((face_masks, faces, pos3s), norms, perVertices, shaded_perVertices) = renderEnvsA(batched_envs, states[i].pipeline_state)
    batched_faces = []
    new_faces = []
    new_pos = []
    for j in range(30): 
        batched_face, face, pos3 = Render.render_by_parts_Filtering(face_masks[j], faces[j], pos3s[j])
        batched_faces.append(batched_face)
        new_faces.append(face)
        new_pos.append(pos3)
    batched_faces= jnp.array(batched_faces)
    pos3s = jnp.array(new_pos)
    faces= jnp.array(new_faces)
    pixels = jax.block_until_ready(jnp.array([renderEnvsC(batched_envs, batched_faces, faces, pos3s, norms, perVertices, shaded_perVertices)]))
    end = time.time_ns()
    frames = jnp.append(frames, pixels, 0)
    print(f"Frame{i} for batch took: {(end - start) / 1000 / 1000} ms")

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
ani.save('./brax_output/animation_with_clipping.gif', writer='pillow', fps=15)