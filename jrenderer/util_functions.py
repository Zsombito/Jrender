from jax import jit
from jaxtyping import Array, Float
import jax.numpy as jnp
from typing import cast
from .r_types import Vec4f, Vec3f

@jit
def normalise(vector: Float[Array, "*a dim"]) -> Float[Array, "*a dim"]:
    """normalise vector in-place."""
    result: Float[Array, "*a dim"] = cast(
        Float[Array, "*a dim"],
        vector / jnp.linalg.norm(vector),
    )
    assert isinstance(result, Float[Array, "*a dim"])

    return result



@jit
def homogenousToCartesian(vector: Vec4f) -> Vec3f:
    """
    Converts homogoneous coordinates to Cartesian
    """
    vec = vector / vector[3]
    return vec[:3]


