from jrenderer.camera import Camera
from jrenderer.object import Model
from jrenderer.scene import Scene
from jrenderer.pipeline import Render
from jrenderer.shader import stdVertexExtractor, stdVertexShader
import jax
import jax.numpy as jnp



vertices = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    [
        [100.000000, -100.000000, 0.000000],
        [100.000000, 100.000000, 0.000000],
        [-100.000000, 100.000000, 0.000000],
        [-100.000000, -100.000000, 0.000000],
    ]
)
vertices = vertices * 0.01
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


model = Model(vertices, normals, indices, uvs)

camera = Camera(
    position=jnp.array([5, 5, 5]) ,
    target=jnp.zeros(3),
    up=jnp.array([0.0, 1.0, 0.0]),
    fov=90,
    aspect=16/9,
    near=0.1,
    far=10000
)

scene = Scene(camera)
idx = scene.add_Model(model)


Render.add_Scene(scene, "MyScene")
Render.geometryStage(stdVertexShader,stdVertexExtractor)