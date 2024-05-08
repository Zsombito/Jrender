from jrenderer.camera import Camera
from jrenderer.object import Model
from jrenderer.scene import Scene
from jrenderer.pipeline import Render
from jrenderer.shader import stdVertexExtractor, stdVertexShader, stdFragmentExtractor, stdFragmentShader
from jrenderer.lights import Light
import jax
import jax.numpy as jnp
import timeit



vertices1 = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    [
        [1.000000, 0.000000, -1.000000],
        [1.000000, 0.000000, 1.000000],
        [-1.000000, 0.000000, 1.000000],
        [-1.000000, 0.000000, -1.000000],
    ]
)
normals = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    [
        [0.000000, 0.000000, 1.000000],
        [0.000000, 0.000000, 1.000000],
        [0.000000, 0.000000, 1.000000],
        [0.000000, 0.000000, 1.000000],
    ]
)

uvs = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    [
        [1.000000, 0.000000],
        [1.000000, 1.000000],
        [0.000000, 1.000000],
        [0.000000, 0.000000],
    ]
)
diffMap = jnp.array([
    [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]
    ]]
)
specMap = jnp.array([
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ]]
)
indices = jnp.array([[0, 2, 3], [0, 1, 2], ])  # pyright: ignore[reportUnknownMemberType]


model1 = Model.create(vertices1, normals, indices, uvs, diffMap, specMap)

camera = Camera(
    position=jnp.array([0, 5, 5]) ,
    target=jnp.zeros(3),
    up=jnp.array([0.0, 1.0, 0.0]),
    fov=90,
    aspect=16/9,
    near=0.1,
    far=10000,
    X=256,
    Y=144
)
light = Light(camera.viewMatrix, [1000, 1000, 1000], [0.0, 150.0, 0.0, 1], 0)
lights = jnp.array([
    light.getJnpArray()])

scene = Scene(camera, lights, 2, 2)
idx = scene.add_Model(model1)
#for i in range(15):
    #print(f"Loop: {i}/10")
    #indices = jnp.append(indices, indices, 0)
#model1 = Model.create(vertices1, normals, indices, uvs)

#idx = scene.add_Model(model1)

scene.changeShader(stdVertexExtractor, stdVertexShader, stdFragmentExtractor, stdFragmentShader)


Render.add_Scene(scene, "MyScene")
Render.render_C()
#Render.render()
#Render.render()
#Render.render()


with jax.profiler.trace("./jax-trace-buffer"):
    frame_buffer = Render.render_C()


from typing import cast

import matplotlib.animation as animation
import matplotlib.figure as figure
import matplotlib.image as mimage
import matplotlib.pyplot as plt

print(frame_buffer.shape)
plt.imshow(jnp.transpose(frame_buffer, [1, 0, 2]))
plt.savefig('output.png')  # pyright: ignore[reportUnknownMemberType]