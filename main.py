from jrenderer.camera import Camera
from jrenderer.model import Model
from jrenderer.scene import Scene
from jrenderer.pipeline import Render
from jrenderer.shader import stdVertexExtractor, stdVertexShader, stdFragmentExtractor, stdFragmentShader
from jrenderer.lights import Light
from jrenderer.capsule import create_capsule
from jrenderer.cube import create_cube
import jax
import jax.numpy as jnp
import timeit



vertices1 = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    [
        [1.000000, 0.000000, -1.000000],
        [1.000000, 0.000000, 1.000000],
        [-1.000000, 0.000000, 1.000000],
        [-1.000000, 0.000000, -1.000000],
    ]
)
normals = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    [
        [0.000000, 0.000000, 1.000000],
        [0.000000, 0.000000, 1.000000],
        [0.000000, 0.000000, 1.000000],
        [0.000000, 0.000000, 1.000000],
    ]
)

uvs = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    [
        [1.000000, 0.000000, 1],
        [1.000000, 1.000000, 1],
        [0.000000, 1.000000, 1],
        [0.000000, 0.000000, 1],
    ]
)
diffMap = jnp.array([
    [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]
    ]]
)
specMap = jnp.array([
    [
        [[0.05, 0.05, 0.05], [0.05, 0.05, 0.05]],
        [[0.05, 0.05, 0.05], [0.05, 0.05, 0.05]]
    ]]
) * 10
indices = jnp.array([[0, 2, 3], [0, 1, 2], ])  # pyright: ignore[reportUnknownMemberType]


model1 = Model(vertices1, normals, indices, uvs, diffMap, specMap)

camera = Camera.create(
    position=jnp.array([-5, 5, 0]) ,
    target=jnp.zeros(3),
    up=jnp.array([0.0, 1.0, 0.0]),
    fov=90,
    aspect=16/9,
    near=0.1,
    far=10000,
    X=1280,
    Y=720
)
light = Light(camera.viewMatrix, [1, 1, 1], [50.0, 150.0, 100.0, 1], 0)
lights = jnp.array([
    light.getJnpArray()])

scene = Scene.create(camera, lights, 2, 2)
for i in range(10):
    print(f"Loop: {i}/10")
    indices = jnp.append(indices, indices, 0)
model1 = Model(vertices1, normals, indices, uvs, diffMap, specMap)

#idx = scene.add_Model(model1)

capsule : Model = create_cube(1.0, diffMap, specMap)
idx, scene =Scene.addModel(scene, model1)



Render.add_Scene(scene, "MyScene")
Render.loadVertexShaders(stdVertexShader, stdVertexExtractor)
Render.loadFragmentShaders(stdFragmentShader, stdFragmentExtractor)

#Render.render()
#Render.render()
#Render.render()

#print(Render.render_with_grad("MyScene")[0].nonzero())



#with jax.profiler.trace("./jax-trace-buffer"):
    #frame_buffer = Render.render_C()



import matplotlib.pyplot as plt

Render.scenes["MyScene"] = Scene.transformModel(Render.scenes["MyScene"], idx, jnp.identity(4, float).at[3,1].set(2))
frame_buffer =Render.render_forward()
print(frame_buffer.shape)
plt.imshow(jnp.transpose(frame_buffer, [1, 0, 2]))
plt.savefig('output.png')  # pyright: ignore[reportUnknownMemberType]

Render.scenes["MyScene"] = Scene.transformModel(Render.scenes["MyScene"], idx, jnp.identity(4, float).at[3,1].set(-2))
with jax.profiler.trace("./jax-trace-full"):
    frame_buffer =Render.render_forward()
plt.imshow(jnp.transpose(frame_buffer, [1, 0, 2]))
plt.savefig('output1.png')  # pyright: ignore[reportUnknownMemberType]