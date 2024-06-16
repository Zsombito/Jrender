from .model import Model
from .camera import Camera
from .r_types import Matrix4, Position, Float, Normal, Integer, Face, UV, TextureMap, Array, Lights
from typing import Tuple, Callable, NamedTuple
import jax.numpy as jnp


class Scene(NamedTuple):
    vertecies : Float[Position, "idx"] 
    normals : Float[Normal, "idx"] 
    uvs : Float[UV, "idx"] 
    modelID : Integer[Array, "idx"]
    modelIDperVertex : Integer[Array, "idx"]
    faces : Integer[Face, "idx"] 
    diffuseText : Float[TextureMap, "idx"]
    specText : Float[TextureMap, "idx"]
    lights : Lights
    mdlMatricies : Float[Array, "* 4 4"]
    unique : int

    @staticmethod
    def create(light, textureX, textureY) -> None:
        """
        Initializes a scene with parameter:
        - light: lights
        - textureX: Texture widths
        - textureY: Texture heights
        """
        vertecies : Float[Position, "idx"] = jnp.empty([0,4], float)
        normals : Float[Normal, "idx"] = jnp.empty([0,4], float)
        uvs : Float[UV, "idx"] = jnp.empty([0,3], float)
        modelID : Integer[Array, "idx"]= jnp.empty([0], int)
        modelIDperVertex : Integer[Array, "idx"]= jnp.empty([0], int)
        faces : Integer[Face, "idx"] = jnp.empty([0,3], int)
        diffuseText = jnp.empty([0, textureX, textureY, 3], float)
        specText = jnp.empty([0, textureX, textureY, 3], float)
        lights = light
        mdlMatricies = jnp.empty([0, 4, 4], float)
        return Scene(vertecies, normals, uvs, modelID, modelIDperVertex, faces, diffuseText, specText, lights, mdlMatricies, 0)
    
    def addModel(self, model : Model):
        """
        Adds a model to the scene
        """
        startIdx = self.vertecies.shape[0]

        changedFaceIdx = jnp.add(model.faces, jnp.ones(model.faces.shape, int) * startIdx)
        faces = jnp.append(self.faces, changedFaceIdx, axis=0)

        vertecies = jnp.append(self.vertecies, model.vertecies, axis=0)
        normals = jnp.append(self.normals, model.normals, axis=0)
        uvs = jnp.append(self.uvs, model.uvs, axis=0)

        newIDs = jnp.ones([model.faces.shape[0]], int) * self.unique
        newIDsforVert = jnp.ones([model.vertecies.shape[0]], int) * self.unique
        modelID = jnp.append(self.modelID, newIDs, axis=0)
        modelIDperVertex = jnp.append(self.modelIDperVertex, newIDsforVert, axis=0)

        diffuseText = jnp.append(self.diffuseText, model.diffuseMap, axis=0)
        specText = jnp.append(self.specText, model.specularMap, axis=0)
        mdlMatricies = jnp.append(self.mdlMatricies, model.mdlMatrix.reshape(1, 4, 4), axis=0)
        unique = self.unique + 1


        return self.unique, Scene(vertecies, normals, uvs, modelID, modelIDperVertex, faces, diffuseText, specText, self.lights, mdlMatricies, unique)
    
    def transformModel(self, idx : int, transform : Matrix4):
        """
        Changes a models moodel Matrix based on the index
        """
        mdlMatricies = self.mdlMatricies.at[idx].set(transform)
        return Scene(self.vertecies, self.normals, self.uvs, self.modelID, self.modelIDperVertex, self.faces, self.diffuseText, self.specText,  self.lights, mdlMatricies, self.unique)
    
    
