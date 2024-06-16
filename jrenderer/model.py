from .r_types import Vec2, Vec3f, Vec3, Vec4f, Position, Face, Normal, TextureMap, UV, Matrix4, Identity4f
from jaxtyping import Array, Float, Integer, UInt8
import jax.numpy as jnp
from typing import Optional, NamedTuple






class Model:
    def __init__(self, vertecies : Vec3f,
                normals : Vec3f, 
                faces : Vec3, 
                uVs : Optional[Vec2] = None, 
                diffuseMap : Optional[TextureMap] = None, 
                specularMap : Optional[TextureMap] = None, 
                transform : Matrix4 = Identity4f
                ) -> None:
        """
        Creates a model instance with the parameters:
        - vertecies: Vertex positions
        - faces: Indecies of 3 vertecies constructing a primitive
        - Uvs: Vertex texture coordinates
        - diffuseMap: Diffuse texture
        - specularMap: Specular texture
        - transform: Model Matrix
        """

        #Mesh Info
        self.vertecies : Float[Position, "idx"] = jnp.apply_along_axis(lambda x : jnp.array([*x, 1.0]), 1, vertecies)
        self.normals : Float[Normal, "idx"] = jnp.apply_along_axis(lambda x : jnp.array([*x, 0.0]), 1, normals) 
        self.faces : Integer[Face, "idx"]= faces

        #Texture info
        self.uvs : Integer[UV, "idx"] = uVs
        self.diffuseMap : Integer[TextureMap, ""] = diffuseMap
        self.specularMap : Integer[TextureMap, ""] = specularMap

        self.mdlMatrix = transform

    


        


