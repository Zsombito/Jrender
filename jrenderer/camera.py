from .r_types import Vec3f, Identity4f, Matrix4, FloatV
from typing import NamedTuple
from .util_functions import normalise
import jax.numpy as jnp
from typing import cast
import jax.lax as lax
from jax import jit
from jaxtyping import Array, Integer, Float

    
class Camera(NamedTuple):
    position : Vec3f
    target : Vec3f
    up : Vec3f
    fov : float
    aspect : float
    near : float
    far : float
    X : int
    Y : int
    viewMatrix : Matrix4
    projection : Matrix4
    transformMatrix : Matrix4
    viewPortMatrix : Matrix4
    pixelsX : Integer[Array, "X"]
    pixelsY : Integer[Array, "Y"]
    defaultFrame : Float[Array, "X Y 3"]

    def create( position : Vec3f, target : Vec3f, up : Vec3f, fov : float, aspect : float, near : float, far : float, X:int , Y: int):
        """
        Creates a Camera instance based on the input parameters
        - position: Position of the camera
        - target: The target of the camera
        - up: The up axis of the camera
        - fov: Field of View
        - aspect: Aspect ratio of the image
        - near: Value for the near clipping plane
        - far: Value for the far clipping plane
        - X: Resolution width
        - Y: Resolution height
        """
        forward : Vec3f = normalise(target - position)
        up = normalise(up)
        side: Vec3f = normalise(jnp.cross(up, forward))
        up : Vec3f = normalise(jnp.cross(forward, side))
        viewMatrix : Matrix4=  (jnp.identity(4)
            .at[:3, 0].set(side)
            .at[:3, 1].set(up)
            .at[:3, 2].set(forward)
            .at[3, 0].set(-jnp.dot(side, position))
            .at[3, 1].set(-jnp.dot(up, position))
            .at[3, 2].set(-jnp.dot(forward, position))
        )
        
        #Perspective Projection Matrix
        f = 1 / jnp.tan(jnp.pi * fov / 360)
        projection: Matrix4 = (jnp.zeros((4,4), float)
            .at[0,0].set(f * aspect)
            .at[1,1].set(f)
            .at[2,2].set(far / (far - near))
            .at[3,2].set(-(far * near) / (far - near))
            .at[2,3].set(1)

        )
        transformMatrix : Matrix4 =  projection @ viewMatrix 


        #Viewport Matrix
        viewPortMatrix = (jnp.identity(4)
            .at[0,0].set(X / 2)
            .at[3,0].set(X / 2)
            .at[1, 1].set(-Y / 2)
            .at[3,1].set(Y / 2))
        
        
        frame_buffer = jnp.zeros((X, Y, 3), float)
        pixelsX, pixelsY = lax.iota(int, X), lax.iota(int, Y)
        return Camera(position, target, up, fov, aspect, near, far, X, Y, viewMatrix, projection, transformMatrix, viewPortMatrix, pixelsX, pixelsY, frame_buffer)

    @jit
    def modify(self, position : Vec3f, target : Vec3f, up : Vec3f):
        zaxis : Vec3f = normalise(target - position)
        xaxis: Vec3f = normalise(jnp.cross(up, zaxis))
        yaxis : Vec3f = jnp.cross(zaxis, xaxis)
        viewMatrix : Matrix4=  (jnp.identity(4)
            .at[:3, 0].set(xaxis)
            .at[:3, 1].set(yaxis)
            .at[:3, 2].set(zaxis)
            .at[3, 0].set(-jnp.dot(xaxis, position))
            .at[3, 1].set(-jnp.dot(yaxis, position))
            .at[3, 2].set(-jnp.dot(zaxis, position))
        )
        return self._replace(viewMatrix = viewMatrix, position = position, target = target, up = up, transformMatrix = self.projection @ viewMatrix)

