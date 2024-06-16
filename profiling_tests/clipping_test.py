from jrenderer.pipeline_brax_without_clipping import Render as Render_without_clip
from jrenderer.pipeline_brax import Render as Render_with_clip
from jrenderer.camera import Camera
from jrenderer.model import Model
from jrenderer.scene import Scene
from jrenderer.shader import stdVertexExtractor, stdVertexShader, stdFragmentExtractor, stdFragmentShader
from jrenderer.lights import Light
from jrenderer.capsule import create_capsule
import jax.numpy as jnp
import jax
import time
import math



Render_with_clip.loadVertexShaders(stdVertexShader, stdVertexExtractor)
Render_without_clip.loadVertexShaders(stdVertexShader, stdVertexExtractor)
Render_with_clip.loadFragmentShaders(stdFragmentShader, stdFragmentExtractor)
Render_without_clip.loadFragmentShaders(stdFragmentShader, stdFragmentExtractor)


camera = Camera.create(
    position=jnp.array([0, 0, 0]) ,
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

scene : Scene = Scene.create(lights, 2, 2)

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


for i in range(20):
    sinx = math.sin(math.pi / 10 * i) 
    cosx = math.cos(math.pi / 10 * i) 
    trans = jnp.identity(4, float).at[3, 0].set(sinx * 10).at[3, 2].set(cosx * 10)
    capsule = create_capsule(1, 1, 1, diffMap, specMap, transform=trans)
    _, scene = scene.addModel(capsule)
    
framesA = []
framesB = []
timesA = []
timesB = []
for i in range(50):
    camera = Camera.create(
        position=jnp.array([0, 0, 0]) ,
        target=jnp.array([math.sin(math.pi/25 * i) * 10, 0, math.cos(math.pi/2.5 * i) * 10], float),
        up=jnp.array([0.0, 1.0, 0.0]),
        fov=90,
        aspect=16/9,
        near=0.1,
        far=10000,
        X=1280,
        Y=720
    )
    start = time.time_ns()
    pixels = jax.block_until_ready(Render_with_clip.render_forward(scene, camera))
    end = time.time_ns()

    framesA.append(jnp.transpose(pixels, [1, 0, 2]).astype("uint8"))
    timesA.append((end - start) / 1000 / 1000)
        
    start = time.time_ns()
    pixels = jax.block_until_ready(Render_without_clip.render_forward(scene, camera))
    end = time.time_ns()

    framesB.append(jnp.transpose(pixels, [1, 0, 2]).astype("uint8"))
    timesB.append((end - start) / 1000 / 1000)


#import imageio
#imageio.mimsave('./outputA.gif', framesA)
#imageio.mimsave('./outputB.gif', framesB)

import matplotlib.pyplot as plt

frameNmb = range(50)

plt.plot(frameNmb, timesA, label="With clip")
plt.plot(frameNmb, timesB, label="Without clip")
plt.legend()
plt.ylabel("Time render (ms)")
plt.xlabel("Index of frame")
plt.suptitle("Clipping test")
plt.savefig("./tests/Clipping_test.png")
plt.savefig("./tests/Clipping_test.svg")

framesA = []
framesB = []
timesA = []
timesB = []
clipping_times = []
for i in range(50):
    camera = Camera.create(
        position=jnp.array([0, 0, 0]) ,
        target=jnp.array([math.sin(math.pi/25 * i) * 10, 0, math.cos(math.pi/2.5 * i) * 10], float),
        up=jnp.array([0.0, 1.0, 0.0]),
        fov=90,
        aspect=16/9,
        near=0.1,
        far=10000,
        X=1280,
        Y=720
    )
    start = time.time_ns()
    pixels, clipping_time= jax.block_until_ready(Render_with_clip.render_forward(scene, camera, debug=True))
    end = time.time_ns()

    framesA.append(jnp.transpose(pixels, [1, 0, 2]).astype("uint8"))
    if i != 0:
        timesA.append((end - start) / 1000 / 1000)
        clipping_times.append(clipping_time)
        
    start = time.time_ns()
    pixels = jax.block_until_ready(Render_without_clip.render_forward(scene, camera))
    end = time.time_ns()

    framesB.append(jnp.transpose(pixels, [1, 0, 2]).astype("uint8"))
    if i != 0:
        timesB.append((end - start) / 1000 / 1000)

frameNmb = range(49)
plt.clf()
plt.plot(frameNmb, timesA, label="With clip")
plt.plot(frameNmb, timesB, label="Without clip")
plt.legend()
plt.ylabel("Time render (ms)")
plt.ylim(0, 150)
plt.xlabel("Index of frame")
plt.suptitle("Clipping test without compilation")
plt.savefig("./tests/Clipping_test_without_compilation.svg")
plt.savefig("./tests/Clipping_test_without_compilation.png")

render_times = []
for i in range(len(timesA)):
    render_times.append(timesA[i] - clipping_times[i])

plt.clf()
plt.plot(frameNmb, clipping_times, label="Time spent filtering")
plt.plot(frameNmb, timesA, label="Time spent rendering")
plt.legend()
plt.ylabel("Time render (ms)")
plt.ylim(0, 80)
plt.xlabel("Index of frame")
plt.suptitle("Filtering and rendering times")
plt.savefig("./tests/Clipping_test_ratio.svg")
plt.savefig("./tests/Clipping_test_ratio.png")