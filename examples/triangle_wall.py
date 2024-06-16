
from jrenderer.camera import Camera
from jrenderer.model import Model
from jrenderer.scene import Scene
from jrenderer.pipeline import Render
from jrenderer.shader import stdVertexExtractor, stdVertexShader, stdFragmentExtractor, stdFragmentShader
from jrenderer.lights import Light
from jrenderer.capsule import create_capsule
import jax
import jax.numpy as jnp
import timeit


vec = jnp.array([
    [-1, 1, 0],
    [1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0]
])

vec2 = jnp.array([
    [-1, 1, -1],
    [1, 1, -1],
    [1, -1, -1],
    [-1, -1, -1]
])

faces = jnp.array([
    [0, 3, 2],
    [0, 1, 2]
])

norm = jnp.array([
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
])

uv = jnp.array([
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
])

diffMap = jnp.array([[[[0, 1, 0]]]])
spec = jnp.array([[[[0.05, 0.05, 0.05]]]]) 
diffMap2 = jnp.array([[[[1, 0, 0]]]])

mdl = Model(vec, norm, faces, uv, diffMap, spec)
mdl2 = Model(vec2, norm, faces, uv, diffMap2, spec)


camera = Camera(
    position=jnp.array([0, 0, 5]) ,
    target=jnp.zeros(3),
    up=jnp.array([0.0, 1.0, 0.0]),
    fov=90,
    aspect=16/9,
    near=0.1,
    far=10000,
    X=1280,
    Y=720
)
light = Light(camera.viewMatrix, [1, 1, 1], [0.0, 0.0, 1.0, 1], 0)
lights = jnp.array([
    light.getJnpArray()])

scene = Scene(camera, lights, 1, 1)
idx = scene.add_Model(mdl)
idx = scene.add_Model(mdl2)
scene.changeShader(stdVertexExtractor, stdVertexShader, stdFragmentExtractor, stdFragmentShader)
Render.add_Scene(scene, "MyScene")
frame_buffer =Render.render_forward()

import matplotlib.pyplot as plt

print(frame_buffer.shape)
plt.imshow(jnp.transpose(frame_buffer, [1, 0, 2]))
plt.savefig('output.png')  # pyright: ignore[reportUnknownMemberType]
