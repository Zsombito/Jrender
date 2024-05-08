from .object import Model
from .camera import Camera
from .r_types import Matrix4, Position, Float, Normal, Integer, Face, UV, TextureMap, Array
from typing import Tuple, Callable
import jax.numpy as jnp


class Scene:
    unique : int = 0


    def __init__(self, camera : Camera, light, textureX, textureY) -> None:
        self.models : dict[int, Tuple[int, int]] = {}
        self.vertecies : Float[Position, "idx"] = jnp.empty([0,4], float)
        self.normals : Float[Normal, "idx"] = jnp.empty([0,4], float)
        self.uvs : Integer[UV, "idx"] = jnp.empty([0,2], float)
        self.modelID : Integer[Array, "idx"]= jnp.empty([0,1], int)
        self.faces : Integer[Face, "idx"] = jnp.empty([0,3], int)
        self.diffuseText = jnp.empty([0, textureX, textureY, 3], float)
        self.specText = jnp.empty([0, textureX, textureY, 3], float)
        self.camera : Camera  = camera
        self.vertexExtractor = None
        self.vertexShader = None
        self.lights = light
        pass

    def add_Model(self, model : Model) -> int:
        startIdx = self.vertecies.shape[0]

        changedFaceIdx = jnp.add(model.faces, jnp.ones(model.faces.shape, int) * startIdx)
        self.faces = jnp.append(self.faces, changedFaceIdx, axis=0)

        self.vertecies = jnp.append(self.vertecies, model.vertecies, axis=0)
        self.normals = jnp.append(self.normals, model.normals, axis=0)
        self.uvs = jnp.append(self.uvs, model.uVs, axis=0)

        newIDs = jnp.ones([model.faces.shape[0],1], int) * Scene.unique
        self.modelID = jnp.append(self.modelID, newIDs, axis=0)
        print(self.modelID)

        self.diffuseText = jnp.append(self.diffuseText, model.diffuseMap, axis=0)
        self.specText = jnp.append(self.specText, model.specularMap, axis=0)

        self.models[Scene.unique] = (startIdx, self.vertecies.shape[0] + 1)
        Scene.unique += 1

        return Scene.unique - 1


    def transform_Model(self, idx : int, transform : Matrix4) -> None:
        startIdx, endIdx = self.models[idx]
        self.vertecies[startIdx:endIdx, :] = self.vertecies[startIdx:endIdx, :] @ transform
        self.normals[startIdx:endIdx, :] = self.normals[startIdx:endIdx, :] @ transform
    
    def changeShader(self, vertexExtractor : Callable, vertexShader : Callable, fragmentExtractor : Callable, fragmentShader : Callable):
        self.vertexShader = vertexShader
        self.vertexExtractor = vertexExtractor
        self.fragmentShader = fragmentShader
        self.fragmentExtractor = fragmentExtractor
        
    
    def delete_Model(self, idx : int) -> None:
        raise Exception("Not implemented yet")
    
        
        
