from jrenderer import Camera, Model, Scene, Render, Light
import jax.numpy as jnp


#Create model parameters
vec = jnp.array([
    [-1, 1, 0],
    [1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0]
])

vec2 = jnp.array([
    [-1, 1, -1],
    [1, 1, -1],
    [1, -1, -1],
    [-1, -1, -1]
])

faces = jnp.array([
    [0, 3, 2],
    [0, 1, 2]
])

norm = jnp.array([
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
])

uv = jnp.array([
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
])

diffMap = jnp.array([[[[0, 1, 0]]]])
spec = jnp.array([[[[0.05, 0.05, 0.05]]]]) 
diffMap2 = jnp.array([[[[1, 0, 0]]]])

#Create Models
mdl = Model(vec, norm, faces, uv, diffMap, spec)
mdl2 = Model(vec2, norm, faces, uv, diffMap2, spec)


#Create camera
camera = Camera.create(
    position=jnp.array([0, 0, 5]) ,
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
light = Light([1, 1, 1], [0.0, 0.0, 1.0, 1], 0)
lights = jnp.array([
    light.getJnpArray()])

#Creat scene
scene : Scene = Scene.create(lights, 1, 1)
idx, scene = scene.addModel(mdl)
idx, scene= scene.addModel(mdl2)


#Render Image
frame_buffer =Render.render_forward(scene, camera).astype("uint8")

import matplotlib.pyplot as plt

print(frame_buffer.shape)
plt.imshow(jnp.transpose(frame_buffer, [1, 0, 2]))
plt.savefig('./brax_output/triangle.png')  
