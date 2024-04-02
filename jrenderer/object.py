from .r_types import Vec2, Vec3f, Vec3, Vec4f, Position, Face, Normal, TextureMap, UV, Matrix4, Identity4f
from jaxtyping import Array, Float, Integer, UInt8
import jax.numpy as jnp
from typing import Optional, NamedTuple






class Model(NamedTuple):
    vertecies : Float[Position, "idx"] 
    normals : Float[Normal, "idx"]
    faces : Integer[Face, "idx"]

    #Texture info
    uVs : Integer[UV, "idx"]
    diffuseMap : Integer[TextureMap, ""] 
    specularMap : Integer[TextureMap, ""]

    @staticmethod
    def create(vertecies : Vec3f,
                normals : Vec3f, 
                faces : Vec3, 
                uVs : Optional[Vec2] = None, 
                diffuseMap : Optional[TextureMap] = None, 
                specularMap : Optional[TextureMap] = None, 
                transform : Matrix4 = Identity4f
                ) -> None:

                
        vertecies4f = jnp.apply_along_axis(lambda x : jnp.array([*x, 1.0]), 1, vertecies)

        #Mesh Info
        v : Float[Position, "idx"] = jnp.matmul(vertecies4f, transform)
        n : Float[Normal, "idx"] = jnp.apply_along_axis(lambda x : jnp.array([*x, 0.0]), 1, normals) @ transform
        f : Integer[Face, "idx"]= faces

        #Texture info
        u : Integer[UV, "idx"] = uVs
        dm : Integer[TextureMap, ""] = diffuseMap
        ds : Integer[TextureMap, ""] = specularMap

        return Model(v, n, f, u, dm, ds)

    def applyTransform(self, transform):
        self.vertecies = jnp.matmul(self.vertecies, transform)
    


        


