from .r_types import Vec2, Vec3f, Vec3, Vec4f, Position, Face, Normal, TextureMap, UV, Matrix4, Identity4f
from jaxtyping import Array, Float, Integer, UInt8
import jax.numpy as jnp
from typing import Optional






class Model:
    def __init__(self, vertecies : Vec3f,
                normals : Vec3f, 
                faces : Vec3, 
                uVs : Optional[Vec2] = None, 
                diffuseMap : Optional[TextureMap] = None, 
                specularMap : Optional[TextureMap] = None, 
                transform : Matrix4 = Identity4f
                ) -> None:

                
        vertecies4f = jnp.apply_along_axis(lambda x : jnp.array([*x, 1.0]), 1, vertecies)

        #Mesh Info
        self.vertecies : Float[Position, "idx"] = jnp.matmul(vertecies4f, transform)
        self.normals : Float[Normal, "idx"] = jnp.apply_along_axis(lambda x : jnp.array([*x, 0.0]), 1, normals) @ transform
        self.faces : Integer[Face, "idx"]= faces

        #Texture info
        self.uVs : Integer[UV, "idx"] = uVs
        self.diffuseMap : Integer[TextureMap, ""] = diffuseMap
        self.specularMap : Integer[TextureMap, ""] = specularMap

    def applyTransform(self, transform):
        self.vertecies = jnp.matmul(self.vertecies, transform)
    


        


