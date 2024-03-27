from jax import jit
from .r_types import Position, Normal, Face, UV, Matrix4, Float, Integer, Array
from .scene import Scene
from typing import Callable
import jax.numpy as jnp

#Standard Vertex Shading (Phong):
def _stdVertexShader(position : Position, normal : Normal, view: Matrix4, proj: Matrix4) -> tuple:
    return (((position @ view) @ proj, (normal @ view) @ proj), None)

stdVertexShader : Callable = jit(_stdVertexShader)
    
#Standard Vertex Info extractor (pos, normal, modelID)
def stdVertexExtractor(scene : Scene):
    pos : Float[Position, "idx"] = jnp.empty([0,4], float)
    norm : Float[Normal, "idx"] = jnp.empty([0,4], float)
    modelID : Integer[Array, "1"]= jnp.empty([0,1], int)
    face : Integer[Face, "idx"] = jnp.empty([0,3], int)
    for idx, model in scene.models.items():
        changedFaceIdx = jnp.add(model.faces, jnp.ones(model.faces.shape, int) * pos.shape[0])
        face = jnp.concatenate((face, changedFaceIdx), axis=0)
        pos = jnp.concatenate((pos, model.vertecies), axis=0)
        norm = jnp.concatenate((norm, model.normals), axis=0)
        newIDs = jnp.ones([model.vertecies.shape[0],1], int) * idx
        modelID = jnp.concatenate((modelID, newIDs), axis=0)
    
    return ((pos, norm, scene.camera.viewMatrix, scene.camera.projection), [0, 0, None, None], (modelID, face))
        
