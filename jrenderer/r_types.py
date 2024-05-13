from jaxtyping import Array, Integer, Float, Bool, UInt8
import jax.numpy as jnp
from typing import Tuple


#Types as arrays
FloatV = Float[Array, ""]
IntegerV = Integer[Array, ""]
BoolV = Bool[Array, ""]

#Vector
Vec2 = Integer[Array, "2"]
Vec3 = Integer[Array, "3"]
Vec2f = Float[Array, "2"]
Vec3f = Float[Array, "3"]
Vec4f = Float[Array, "4"]

Color = UInt8[Array, "3"]

#Matricies
Matrix4 = Float[Array, "4 4"]
Matrix3 = Float[Array, "3 3"]

#Model related
Position = Float[Vec4f, ""]
Face = Integer[Vec3, ""]
Normal = Float[Vec4f, ""]
UV = Float[Vec3f, ""]
TextureMap = Float[Array, "X Y 3"]

Lights = Float[Array, "7"]


#Combined Type
PosXNorm = Tuple[Float[Position, "idx"], Float[Normal, "idx"]]


#Special instances
Identity4f = jnp.identity(4, jnp.float32)




