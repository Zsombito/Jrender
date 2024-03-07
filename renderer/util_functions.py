from jax import jit
from jaxtyping import Array, Float
import jax.numpy as jnp
from typing import cast

def _normalise(vector: Float[Array, "*a dim"]) -> Float[Array, "*a dim"]:
    """normalise vector in-place."""
    result: Float[Array, "*a dim"] = cast(
        Float[Array, "*a dim"],
        vector / jnp.linalg.norm(vector),
    )
    assert isinstance(result, Float[Array, "*a dim"])

    return result

normalise : function = jit(_normalise)

