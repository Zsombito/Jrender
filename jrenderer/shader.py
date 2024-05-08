from jax import jit, vmap
from .r_types import Position, Normal, Face, UV, Matrix4, Float, Integer, Array
from .scene import Scene
from .object import Model
from .util_functions import normalise
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
    

    return ((pos, norm, scene.camera.viewMatrix, scene.camera.projection), [0, 0, None, None], face, (modelID, pos[:, :3]))


@jit
def stdFragmentExtractor(idx, faces, norm, perVertexExtra, shaded_perVertexExtra):
    face = faces[idx]
    v1, v2, v3 = face[0], face[1], face[2]
    normal = jnp.array([norm[v1], norm[v2], norm[v3]])
    modelID = perVertexExtra[0][idx]
    worldSpacePosition = jnp.array([perVertexExtra[1][v1], perVertexExtra[1][v2], perVertexExtra[1][v3]])
    
    
    return [normal, worldSpacePosition]

@jit
def stdFragmentShader(interpolatedFrag, lights, normals, worldSpacePosition):
    def perLight(light, pos, norm):
        kspec = jnp.array([0.05, 0.05, 0.05])
        kdiff = jnp.array([0.8, 0.8, 0.8])
        I = light[3:6]  / jnp.where(light[6] == 0, 1.0, (jnp.linalg.norm(light[:3] - pos)) * (jnp.linalg.norm(light[:3] - pos)))
        V =  normalise(-pos)
        L = normalise(pos - light[:3])
        R = normalise(L - 2 * (jnp.dot(L, norm) * norm))
        Ispec = I * (kspec * jnp.dot(R, V) ** 32)
        Idiff = I * (jnp.dot(L, norm) * kdiff)
        return jnp.clip(Ispec + Idiff, 0, 1)




    alpha = interpolatedFrag[0]
    beta = interpolatedFrag[1]
    gamma = interpolatedFrag[2]

    pos = worldSpacePosition[0] * alpha + worldSpacePosition[1] * beta + worldSpacePosition[2] * gamma
    normal = normalise(normals[0] * alpha + normals[1] * beta + normals[2] * gamma)
    normal = normal[:3]
    color = vmap(perLight, [0, None, None])(lights, pos, normal)
    color = jnp.clip(color.sum(0), 0, 1)
    return jnp.array([interpolatedFrag[4], color[0], color[1], color[2]])

        
