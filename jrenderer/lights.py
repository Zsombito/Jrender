from .util_functions import homogenousToCartesian
import jax.numpy as jnp






class Light:
    def __init__(self, viewMatrix, intenisity, position, point):
        self.view = viewMatrix
        self.intensity = intenisity
        self.pos = jnp.array(position)
        self.point = point
    
    def getJnpArray(self):
        pos = self.pos[:3]
        return jnp.array([pos[0], pos[1], pos[2], self.intensity[0], self.intensity[1], self.intensity[2], self.point])
        
        