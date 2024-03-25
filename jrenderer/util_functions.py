from jax import jit
from jaxtyping import Array, Float
import jax.numpy as jnp
from typing import cast
from .r_types import Vec4f, Vec3f

def _normalise(vector: Float[Array, "*a dim"]) -> Float[Array, "*a dim"]:
    """normalise vector in-place."""
    result: Float[Array, "*a dim"] = cast(
        Float[Array, "*a dim"],
        vector / jnp.linalg.norm(vector),
    )
    assert isinstance(result, Float[Array, "*a dim"])

    return result

normalise = jit(_normalise)


def _homogenousToCartesian(vector: Vec4f) -> Vec3f:
    vec = vector / vector[3]
    return vec[:3]

homogenousToCartesian = jit(_homogenousToCartesian)

