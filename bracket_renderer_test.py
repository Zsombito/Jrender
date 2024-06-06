from jrenderer.camera import Camera
from jrenderer.model import Model
from jrenderer.scene import Scene
from jrenderer.pipeline import Render
from jrenderer.shader import stdVertexExtractor, stdVertexShader, stdFragmentExtractor, stdFragmentShader
from jrenderer.lights import Light
from jrenderer.capsule import create_capsule
from jrenderer.cube import create_cube
import jax
import jax.numpy as jnp
import time






Render.loadVertexShaders(stdVertexShader, stdVertexExtractor)
Render.loadFragmentShaders(stdFragmentShader, stdFragmentExtractor)

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
        [1.000000, 0.000000, 1],
        [1.000000, 1.000000, 1],
        [0.000000, 1.000000, 1],
        [0.000000, 0.000000, 1],
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
        [[0.05, 0.05, 0.05], [0.05, 0.05, 0.05]],
        [[0.05, 0.05, 0.05], [0.05, 0.05, 0.05]]
    ]]
) * 10

camera = Camera.create(
    position=jnp.array([-2, 2, 0]) ,
    target=jnp.zeros(3),
    up=jnp.array([0.0, 1.0, 0.0]),
    fov=90,
    aspect=16/9,
    near=0.1,
    far=10000,
    X=1280,
    Y=720
)
light = Light(camera.viewMatrix, [1, 1, 1], [50.0, 150.0, 100.0, 1], 0)
lights = jnp.array([
    light.getJnpArray()])

scene = Scene.create(lights, 2, 2)

capsule : Model = create_capsule(0.5, 1, 1, diffMap, specMap)
idx, scene =Scene.addModel(scene, capsule)


Render.add_Scene(scene, "Test")



import matplotlib.pyplot as plt

_, frame_buffer = Render.render_forward(scene, camera)



xs = []
ys = []
for i in range(1, 10):
    camera = Camera.create(
        position=jnp.array([-2 - (10 - i) * 0.5, 2 + (10 - i) * 0.5, 0]) ,
        target=jnp.zeros(3),
        up=jnp.array([0.0, 1.0, 0.0]),
        fov=90,
        aspect=16/9,
        near=0.1,
        far=10000,
        X=1280,
        Y=720
    )
    start = time.time_ns()
    nums, pixel =jax.block_until_ready(Render.render_forward(scene, camera))
    end = time.time_ns()
    xs.append(nums)
    ys.append((end - start) / 1000 / 1000)

    
xs = []
ys = []
for i in range(1, 10):
    camera = Camera.create(
        position=jnp.array([-2 - (10 - i) * 0.5, 2 + (10 - i) * 0.5, 0]) ,
        target=jnp.zeros(3),
        up=jnp.array([0.0, 1.0, 0.0]),
        fov=90,
        aspect=16/9,
        near=0.1,
        far=10000,
        X=1280,
        Y=720
    )
    avg = 0
    for _ in range(10):
        start = time.time_ns()
        nums, pixel =jax.block_until_ready(Render.render_forward(scene, camera))
        end = time.time_ns()
        avg += (end - start) / 1000 / 1000
    xs.append(nums)
    ys.append((avg / 10) )

plt.plot(xs, ys)
plt.ylabel("Time taken to render image (ms)")
plt.xlabel("Number of brackets to render")
plt.suptitle("Bracketing test")
plt.savefig("./tests/BrackitingTest.png")
plt.savefig("./tests/BrackitingTest.svg")






