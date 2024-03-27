from .r_types import Vec3f, Identity4f, Matrix4, FloatV
from .util_functions import normalise
import jax.numpy as jnp
from typing import cast
import jax.lax as lax
from jax import jit


class Camera:
    def __init__(self, position : Vec3f, target : Vec3f, up : Vec3f, fov : float, aspect : float, near : float, far : float) -> None:
        self.position = position
        self.target = target
        self.up = up
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        self.updateMatricies()


    def updateMatricies(self) -> None:

        #View matrix
        forward : Vec3f = normalise(self.target - self.position)
        up = normalise(self.up)
        side: Vec3f = normalise(jnp.cross(up, forward))
        up : Vec3f = normalise(jnp.cross(forward, side))
        self.viewMatrix : Matrix4=  (jnp.identity(4)
            .at[:3, 0].set(side)
            .at[:3, 1].set(up)
            .at[:3, 2].set(forward)
            .at[3, 0].set(-jnp.dot(side, self.position))
            .at[3, 1].set(-jnp.dot(up, self.position))
            .at[3, 2].set(-jnp.dot(forward, self.position))
        )
        
        #Perspective Projection Matrix
        f = 1 / jnp.tan(jnp.pi * self.fov / 360)
        self.projection: Matrix4 = (jnp.zeros((4,4), float)
            .at[0,0].set(f * self.aspect)
            .at[1,1].set(f)
            .at[2,2].set(self.far / (self.far - self.near))
            .at[3,2].set(-(self.far * self.near) / (self.far - self.near))
            .at[2,3].set(1)

        )

        self.transformMatrix : Matrix4 =  self.projection @ self.viewMatrix 

    
    
