from jrenderer.pipeline_brax_without_clipping import Render as Render_without_clip
from jrenderer.pipeline_brax import Render as Render_with_clip
from jrenderer.camera import Camera
from jrenderer.model import Model
from jrenderer.scene import Scene
from jrenderer.shader import stdVertexExtractor, stdVertexShader, stdFragmentExtractor, stdFragmentShader
from jrenderer.lights import Light
import jax.numpy as jnp



Render_with_clip.loadVertexShaders(stdVertexShader, stdVertexExtractor)
Render_with_clip.loadVertexShaders(stdVertexShader, stdVertexExtractor)
Render_without_clip.loadFragmentShaders(stdFragmentShader, stdFragmentExtractor)
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

scene = Scene.create(lights, 2, 2)
