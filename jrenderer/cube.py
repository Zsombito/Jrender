
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]
from jrenderer.object import Model


with jax.ensure_compile_time_eval():
    _verts = jnp.array(  # pyright: ignore[reportUnknownMemberType]
        (
            # back
            (-1.0, -1.0, 1.0),  # 0
            (1.0, -1.0, 1.0),  # 1
            (1.0, 1.0, 1.0),  # 2
            (-1.0, 1.0, 1.0),  # 3
            # front
            (-1.0, -1.0, -1.0),  # 4
            (1.0, -1.0, -1.0),  # 5
            (1.0, 1.0, -1.0),  # 6
            (-1.0, 1.0, -1.0),  # 7
            # right
            (-1.0, -1.0, -1.0),  # 8
            (-1.0, 1.0, -1.0),  # 9
            (-1.0, 1.0, 1.0),  # 10
            (-1.0, -1.0, 1.0),  # 11
            # left
            (1.0, -1.0, -1.0),  # 12
            (1.0, 1.0, -1.0),  # 13
            (1.0, 1.0, 1.0),  # 14
            (1.0, -1.0, 1.0),  # 15
            # bottom
            (-1.0, -1.0, -1.0),  # 16
            (-1.0, -1.0, 1.0),  # 17
            (1.0, -1.0, 1.0),  # 18
            (1.0, -1.0, -1.0),  # 19
            # top
            (-1.0, 1.0, -1.0),  # 20
            (-1.0, 1.0, 1.0),  # 21
            (1.0, 1.0, 1.0),  # 22
            (1.0, 1.0, -1.0),  # 23
        )
    )
    _normals  = jnp.array(  # pyright: ignore[reportUnknownMemberType]
        (
            # back
            (0.0, 0.0, 1.0),  # 0
            (0.0, 0.0, 1.0),  # 1
            (0.0, 0.0, 1.0),  # 2
            (0.0, 0.0, 1.0),  # 3
            # front
            (0.0, 0.0, -1.0),  # 4
            (0.0, 0.0, -1.0),  # 5
            (0.0, 0.0, -1.0),  # 6
            (0.0, 0.0, -1.0),  # 7
            # right
            (-1.0, 0.0, 0.0),  # 8
            (-1.0, 0.0, 0.0),  # 9
            (-1.0, 0.0, 0.0),  # 10
            (-1.0, 0.0, 0.0),  # 11
            # left
            (1.0, 0.0, 0.0),  # 12
            (1.0, 0.0, 0.0),  # 13
            (1.0, 0.0, 0.0),  # 14
            (1.0, 0.0, 0.0),  # 15
            # bottom
            (0.0, -1.0, 0.0),  # 16
            (0.0, -1.0, 0.0),  # 17
            (0.0, -1.0, 0.0),  # 18
            (0.0, -1.0, 0.0),  # 19
            # top
            (0.0, 1.0, 0.0),  # 20
            (0.0, 1.0, 0.0),  # 21
            (0.0, 1.0, 0.0),  # 22
            (0.0, 1.0, 0.0),  # 23
        )
    )
    _uvs = jnp.array(  # pyright: ignore[reportUnknownMemberType]
        (
            (0.75, 0.25, 1),  # 0
            (1, 0.25, 1),  # 1
            (1, 0, 1),  # 2
            (0.75, 0, 1),  # 3
            # front
            (0.5, 0.25, 1),  # 4
            (0.25, 0.25, 1),  # 5
            (0.25, 0, 1),  # 6
            (0.5, 0, 1),  # 7
            # right
            (0.5, 0, 1),  # 8
            (0.75, 0, 1),  # 9
            (0.75, 0.25, 1),  # 10
            (0.5, 0.25, 1),  # 11
            # left
            (0.25, 0.5, 1),  # 12
            (0.25, 0.25, 1),  # 13
            (0, 0.25, 1),  # 14
            (0, 0.5, 1),  # 15
            # bottom
            (0.25, 0.5, 1),  # 16
            (0.25, 0.25, 1),  # 17
            (0.5, 0.25, 1),  # 18
            (0.5, 0.5, 1),  # 19
            # top
            (0, 0, 1),  # 20
            (0, 0.25, 1),  # 21
            (0.25, 0.25, 1),  # 22
            (0.25, 0, 1),  # 23
        )
    )
    _faces = jnp.array(  # pyright: ignore[reportUnknownMemberType]
        (
            # back face
            (0, 1, 2),
            (0, 2, 3),
            # front face
            (6, 5, 4),
            (7, 6, 4),
            # right face
            (10, 9, 8),
            (11, 10, 8),
            # left face
            (12, 13, 14),
            (12, 14, 15),
            # bottom face
            (18, 17, 16),
            (19, 18, 16),
            # top face
            (20, 21, 22),
            (20, 22, 23),
        )
    )


def create_cube(
    half_extents: Float[Array, "3"],
    diffuse_map,
    specular_map,
):
    return Model.create(_verts * half_extents, _normals, _faces, _uvs * 4, diffuse_map, specular_map)
    return Model(
        verts=_verts * half_extents,
        norms=_normals,
        uvs=_uvs * 4,
        faces=_faces,
        faces_norm=_faces,
        faces_uv=_faces,
        diffuse_map=diffuse_map,
        specular_map=specular_map,
    )
