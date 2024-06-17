
from jrenderer import Scene, Camera, Light, create_cube
from jrenderer import Render_with_Clip as Render
import jax.numpy as jnp


#Diff Texture
diffMap = jnp.array([
    [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]
    ]]
)

#Specular Map
specMap = jnp.array([
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ]]
)

#Creating camera
camera : Camera = Camera.create(
    position=jnp.array([10, 10, 10]) ,
    target=jnp.zeros(3),
    up=jnp.array([0.0, 1.0, 0.0]),
    fov=90,
    aspect=1,
    near=0.1,
    far=10000,
    X=32,
    Y=32
)

#Create Lights
light = Light([50, 50, 50], [5.0, 5.0, 5.0], 1)
lights = jnp.array([
    light.getJnpArray()])

#Create Scene
scene : Scene = Scene.create(lights, 4, 4)

#Create and add cube to the scene
cubeMdl = create_cube(2, diffMap, specMap) 
_, scene = scene.addModel(cubeMdl)

#Gradients
gradients = Render.render_gradients(scene, camera)