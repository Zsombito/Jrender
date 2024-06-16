import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]
from .model import Model
from .util_functions import homogenousToCartesian
from .r_types import Vec3f


def create_plane(size : Vec3f, diffText, specText):
    """
    Creates a plane:
    - size: Size of the plane
    - diffText: Diffuse texture
    - specText: Specular texture
    """
    vertex = jnp.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0]
    ])
    normal = jnp.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ])
    uv = jnp.array([
         [0, 0, 1],
         [0, 0.5, 1],
         [0, 0.5, 1],
         [0, 1, 1],
    ])
    face = jnp.array([
        [0, 1, 2],
        [1, 2, 3]
    ])
    vertecies = jnp.empty((0, 3), float)
    normals = jnp.empty((0, 3), float)
    uvs = jnp.empty((0, 3), float)
    faces = jnp.empty((0, 3), int)
   
   
    for i in range(-5, 5):
        for j in range(-5, 5):
            transition = jnp.array([
                [i, j, 0],
                [i, j, 0],
                [i, j, 0],
                [i, j, 0]
                ])
            len_vertecies = vertecies.shape[0]
            vertecies = jnp.append(vertecies, vertex + transition, 0)
            uvs = jnp.append(uvs, uv, 0)
            faces = jnp.append(faces, face + len_vertecies, 0)
            normals = jnp.append(normals, normal, 0)
    
    return Model(vertecies * 10, normals, faces, uvs, diffText, specText)

   

        

   
   
