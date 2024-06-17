
from jrenderer import Render, Scene, Camera, Light, create_cube
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
    aspect=16/9,
    near=0.1,
    far=10000,
    X=1280,
    Y=720
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

#Generate Image
frame_buffer =Render.render_forward(scene, camera).astype("uint8")


import matplotlib.pyplot as plt

print(frame_buffer.shape)
plt.imshow(jnp.transpose(frame_buffer, [1, 0, 2]))
plt.savefig('./brax_output/cube.png')  # pyright: ignore[reportUnknownMemberType]
