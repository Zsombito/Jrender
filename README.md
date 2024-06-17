[![PyPI Version](https://img.shields.io/pypi/v/jrenderer?logo=pypi)](https://pypi.org/project/jrenderer)
[![Python Versions](https://img.shields.io/pypi/pyversions/jrenderer?logo=python)](https://pypi.org/project/jrenderer)
[![License](https://img.shields.io/github/license/JoeyTeng/jrenderer)](https://github.com/JoeyTeng/jrenderer/blob/master/LICENSE)



# Jrender: A Differentiable Batch Renderer in JAX

Jrenderer is a differentiable batch renderer written in [Jax](https://github.com/google/jax), which can utilize both GPU and TPU accelerators. The GitHub repo also includes [Brax](https://github.com/google/brax/) integration, therefore it can be used to replace the inbuilt rendering capabilities of Brax.

## Installation

The Jrenderer software package is available on [PyPI](https://pypi.org/project/jrenderer), and can be installed with: 

```bash
pip install jrenderer
```

Note: Jax installation has to be done separately if you want to use GPU / TPU accelerators. Please refer to [JAX's installation guide](https://github.com/google/jax#installation) for more details.

## Usage

The following simple example renders a cube with a diffuse texture.
```python
from jrenderer import Render, Scene, Camera, Light, create_cube
import jax.numpy as jnp


#Diff Texture
diffMap = jnp.array([
    [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]
    ]]
)

#Specular Map
specMap = jnp.array([
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ]]
)

#Creating camera
camera : Camera = Camera.create(
    position=jnp.array([10, 10, 10]) ,
    target=jnp.zeros(3),
    up=jnp.array([0.0, 1.0, 0.0]),
    fov=90,
    aspect=16/9,
    near=0.1,
    far=10000,
    X=1280,
    Y=720
)

#Create Lights
light = Light([50, 50, 50], [5.0, 5.0, 5.0], 1)
lights = jnp.array([
    light.getJnpArray()])

#Create Scene
scene : Scene = Scene.create(lights, 4, 4)

#Create and add cube to the scene
cubeMdl = create_cube(2, diffMap, specMap) 
_, scene = scene.addModel(cubeMdl)

#Generate Image
frame_buffer =Render.render_forward(scene, camera).astype("uint8")
```

For more complex examples check [examples](examples) or the [profiling tests](profiling_tests) included in the repo.

## Shader Support

The renderer currently supports a standard shader package, which implements Phong shading.

To implement your own shaders, check [shader.py](jrenderer/shader.py). Implementing a shader is as easy as writing your own jitted shader functions. Once written it can be applied with the following code:
```python
Render.loadVertexShaders(myVertexShader, myVertexExtractor)
Render.loadFragmentShaders(myFragmentShader, myFragmentExtractor)
```
