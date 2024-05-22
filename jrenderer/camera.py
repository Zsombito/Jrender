from .r_types import Vec3f, Identity4f, Matrix4, FloatV
from typing import NamedTuple
from .util_functions import normalise
import jax.numpy as jnp
from typing import cast
import jax.lax as lax
from jax import jit
from jaxtyping import Array, Integer, Float


#class Camera:
    #def __init__(self, position : Vec3f, target : Vec3f, up : Vec3f, fov : float, aspect : float, near : float, far : float, X:int , Y: int) -> None:
        #self.position = position
        #self.target = target
        #self.up = up
        #self.fov = fov
        #self.aspect = aspect
        #self.near = near
        #self.far = far
        #self.X = X
        #self.Y = Y
        #self.updateMatricies()
    


    #def updateMatricies(self) -> None:

        ##View matrix
        #forward : Vec3f = normalise(self.target - self.position)
        #up = normalise(self.up)
        #side: Vec3f = normalise(jnp.cross(up, forward))
        #up : Vec3f = normalise(jnp.cross(forward, side))
        #self.viewMatrix : Matrix4=  (jnp.identity(4)
            #.at[:3, 0].set(side)
            #.at[:3, 1].set(up)
            #.at[:3, 2].set(forward)
            #.at[3, 0].set(-jnp.dot(side, self.position))
            #.at[3, 1].set(-jnp.dot(up, self.position))
            #.at[3, 2].set(-jnp.dot(forward, self.position))
        #)
        
        ##Perspective Projection Matrix
        #f = 1 / jnp.tan(jnp.pi * self.fov / 360)
        #self.projection: Matrix4 = (jnp.zeros((4,4), float)
            #.at[0,0].set(f * self.aspect)
            #.at[1,1].set(f)
            #.at[2,2].set(self.far / (self.far - self.near))
            #.at[3,2].set(-(self.far * self.near) / (self.far - self.near))
            #.at[2,3].set(1)

        #)
        #self.transformMatrix : Matrix4 =  self.projection @ self.viewMatrix 


        ##Viewport Matrix
        #self.viewPortMatrix = (jnp.identity(4)
            #.at[0,0].set(self.X / 2)
            #.at[3,0].set(self.X / 2)
            #.at[1, 1].set(-self.Y / 2)
            #.at[3,1].set(self.Y / 2))

    
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

