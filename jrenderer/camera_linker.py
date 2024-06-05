
from .r_types import Vec3f, Identity4f, Matrix4, FloatV
from typing import NamedTuple
from .util_functions import normalise
from .camera import Camera
import jax.numpy as jnp
from typing import cast
import jax.lax as lax
from jax import jit
from jaxtyping import Array, Integer, Float
from brax import math

class CameraLink(NamedTuple):
    mode : int
    target_idx : int

    def updateCamera(self, camera : Camera, xpos, xrot, geom_pos, geom_rot, geom_link_idx) -> Camera:
        link_idx = geom_link_idx[self.target_idx]
        pos = xpos[link_idx]
        rot = xrot[link_idx]
        target_pos = pos + math.rotate(geom_pos[self.target_idx], rot)
        target_rot = math.quat_mul(rot, geom_rot[self.target_idx])
        pos =  jnp.where(self.mode <= 1, camera.position, jnp.where(
            self.mode == 2, target_pos + jnp.array([0, -5, 1]), target_pos)
        )

        up = camera.up

        target = jnp.where(self.mode == 0, camera.target, jnp.where(
            self.mode <= 2, target_pos, math.rotate(jnp.array([1, 0, 0]), target_rot)
        ))
        return camera.modify(pos, target, up)

