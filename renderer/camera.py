from .r_types import Vec3f, Identity4f, Matrix4, FloatV
from .util_functions import normalise
import jax.numpy as jnp
from typing import cast
import jax.lax as lax


class Camera:
    def __init__(self, position : Vec3f, target : Vec3f, up : Vec3f, horizontalFoV : float, aspect : float, near : float, far : float) -> None:
        self.position = position
        self.target = target
        self.up = up
        self.horizontalFoV = horizontalFoV
        self.aspect = aspect
        self.near = near
        self.far = far
        self.updateMatricies()


    def updateMatricies(self) -> None:

        #View matrix
        forward : Vec3f = normalise(self.position - self.target)
        up = normalise(self.up)
        side: Vec3f = normalise(jnp.cross(forward, up))
        up = cast(Vec3f, jnp.cross(side, forward))
        self.viewMatrix : Matrix4= (jnp.identity(4, float)
            .at[0, :3].set(side)
            .at[1, :3].set(up)
            .at[2, :3].set(-forward)) @ jnp.identity(4).at[:3, 3].set(-self.position)
        
        
        #Perspective Projection Matrix
        deg: float = jnp.asarray(fovy, dtype=jnp.single)  # pyright: ignore
        f: float = 1.0 / lax.tan(  # pyright: ignore[reportUnknownMemberType]
            cast(float, jnp.radians(deg) / 2.0)
        )
        self.projection: Matrix4 = (
            jnp.zeros((4, 4), dtype=jnp.single)  # pyright: ignore
            .at[0, 0]
            .set(f / self.aspect)
            .at[1, 1]
            .set(f)
            .at[2, 2]
            .set((self.far + self.near) / (self.near - self.far))
            # translate z
            .at[2, 3]
            .set((2.0 * self.far * self.near) / (self.far - self.near))
            .at[3, 2]
            .set(-1.0)  # let \omega be -z
        )
