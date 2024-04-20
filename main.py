from jrenderer.camera import Camera
from jrenderer.object import Model
from jrenderer.scene import Scene
from jrenderer.pipeline import Render
from jrenderer.shader import stdVertexExtractor, stdVertexShader
import jax
import jax.numpy as jnp
import timeit



vertices = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    [
        [100.000000, -100.000000, 0.000000],
        [100.000000, 100.000000, 0.000000],
        [-100.000000, 100.000000, 0.000000],
        [-100.000000, -100.000000, 0.000000],
    ]
)
vertices2 = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    [
        [100, 200, 0],
        [200, 200, 0],
        [200, 100, 0],
        [100, 100, 0],
    ]
)
vertices1 = vertices * 0.01
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
indices = jnp.array([[0, 1, 2], [0, 2, 3]])  # pyright: ignore[reportUnknownMemberType]


model1 = Model.create(vertices1, normals, indices, uvs)
model2 = Model.create(vertices2, normals, indices, uvs)

camera = Camera(
    position=jnp.array([5, 5, 5]) ,
    target=jnp.zeros(3),
    up=jnp.array([0.0, 1.0, 0.0]),
    fov=90,
    aspect=16/9,
    near=0.1,
    far=10000,
    X=256,
    Y=144
)

scene = Scene(camera)
idx = scene.add_Model(model1)
idx = scene.add_Model(model2)
for i in range(12):
    print(f"Loop: {i}/10")
    indices = jnp.append(indices, indices, 0)
model1 = Model.create(vertices1, normals, indices, uvs)

idx = scene.add_Model(model1)

scene.changeShader(stdVertexExtractor, stdVertexShader)


Render.add_Scene(scene, "MyScene")
Render.render()
#Render.render()
#Render.render()
#Render.render()


with jax.profiler.trace("./jax-trace-minibatching"):
    Render.render()
