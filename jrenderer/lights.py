import jax.numpy as jnp






#Basic wrepper for lights
class Light:
    def __init__(self, intenisity, position, point):
        """
        Creates a light which can be acces via getJnpArray()
        Parameters:
        - intensity: [R, G, B]
        - position: [X, Y, Z]
        - point: 0 = directional, 1 = point
        """
        self.intensity = jnp.array(intenisity)
        self.pos = jnp.append(jnp.array(position), jnp.array([1]), 0)
        self.point = point
    
    def getJnpArray(self):
        pos = self.pos[:3]
        return jnp.array([pos[0], pos[1], pos[2], self.intensity[0], self.intensity[1], self.intensity[2], self.point])
        
        