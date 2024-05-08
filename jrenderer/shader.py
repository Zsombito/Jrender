from jax import jit, vmap
from .r_types import Position, Normal, Face, UV, Matrix4, Float, Integer, Array
from .scene import Scene
from .object import Model
from .util_functions import normalise, homogenousToCartesian
from typing import Callable
import jax.numpy as jnp

#Standard Vertex Shading (Phong):
@jit
def stdVertexShader(position : Position, normal : Normal, uv,  view: Matrix4, proj: Matrix4) -> tuple:
    posProj = position @ view @ proj
    uv = uv / posProj[3]
    return ((posProj, (normal @ view) @ proj), [position[:3], uv])


    
#Standard Vertex Info extractor (pos, normal, modelID)
def stdVertexExtractor(scene : Scene):
    pos : Float[Position, "idx"] = scene.vertecies
    norm : Float[Normal, "idx"] = scene.normals
    modelID : Integer[Array, "1"]= scene.modelID
    face : Integer[Face, "idx"] = scene.faces
    uv = scene.uvs
    

    return ((pos, norm, uv, scene.camera.viewMatrix, scene.camera.projection), [0, 0, 0, None, None], face, [modelID])


@jit
def stdFragmentExtractor(idx, faces, norm, perVertexExtra, shaded_perVertexExtra):
    face = faces[idx]
    v1, v2, v3 = face[0], face[1], face[2]
    normal = jnp.array([norm[v1], norm[v2], norm[v3]])
    worldSpacePosition = jnp.array([shaded_perVertexExtra[0][v1], shaded_perVertexExtra[0][v2], shaded_perVertexExtra[0][v3]])
    modelID = perVertexExtra[0][idx]
    uv = jnp.array([shaded_perVertexExtra[1][v1], shaded_perVertexExtra[1][v2], shaded_perVertexExtra[1][v3]])
    
    
    return [normal, worldSpacePosition, uv] , modelID

@jit
def stdFragmentShader(interpolatedFrag, lights, diffText, specText, normals, worldSpacePosition, uvs):
    def perLight(light, pos, norm, kdiff, kspec):
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
    uv = uvs[0] *  alpha + uvs[1] * beta + uvs[2]  * gamma
    uv = jnp.round(jnp.array([uv[0] / uv[2], uv[1] / uv[2]]), 0)
    uv = uv.astype(int)
    
    kdiff = diffText[uv[0] , uv[1], :]
    kspec = specText[uv[0], uv[1], :]
    normal = normalise(normals[0] * alpha + normals[1] * beta + normals[2] * gamma)
    normal = normal[:3]
    color = vmap(perLight, [0, None, None, None, None])(lights, pos, normal, kdiff, kspec)
    color = jnp.clip(color.sum(0), 0, 1)
    return jnp.array([interpolatedFrag[4], color[0], color[1], color[2]])

        
