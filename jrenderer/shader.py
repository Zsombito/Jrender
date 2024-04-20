from jax import jit
from .r_types import Position, Normal, Face, UV, Matrix4, Float, Integer, Array
from .scene import Scene
from .object import Model
from typing import Callable
import jax.numpy as jnp

#Standard Vertex Shading (Phong):
@jit
def stdVertexShader(position : Position, normal : Normal, view: Matrix4, proj: Matrix4) -> tuple:
    return (((position @ view) @ proj, (normal @ view) @ proj), None)


    
#Standard Vertex Info extractor (pos, normal, modelID)
def stdVertexExtractor(scene : Scene):
    pos : Float[Position, "idx"] = scene.vertecies
    norm : Float[Normal, "idx"] = scene.normals
    modelID : Integer[Array, "1"]= scene.modelID
    face : Integer[Face, "idx"] = scene.faces

    return ((pos, norm, scene.camera.viewMatrix, scene.camera.projection), [0, 0, None, None], face, modelID)

        
