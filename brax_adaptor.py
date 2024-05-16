
from typing import Iterable, NamedTuple, Optional

import jax
from jax import numpy as jnp
import numpy as onp

import brax
from brax import base, envs, math, positional
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model 


import trimesh

from jrenderer.camera import Camera
from jrenderer.lights import Light
from jrenderer.capsule import create_capsule
from jrenderer.cube import create_cube
from jrenderer.pipeline_without_filtering import Render
from jrenderer.model import Model
from jrenderer.scene import Scene
from jrenderer.shader import stdVertexExtractor, stdVertexShader, stdFragmentExtractor, stdFragmentShader
import mujoco
from mujoco import mjx
import mediapy as media

import mujoco.testdata

import time


def _build_scene(sys: brax.System) -> tuple[Scene, dict[int, int]]:
    
    
    pass
    
xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_box" pos=".2 .2 .2" size=".1 .1 .1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# Make model, data, and renderer
mj_model = mujoco.MjModel.from_xml_string(xml)
mj_data = mujoco.MjData(mj_model)
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)



def _getCamera():
    camera = Camera.create(
        position=jnp.array([2, 2, 2]) ,
        target=jnp.zeros(3),
        up=jnp.array([1.0, 0.0, 0.0]),
        fov=90,
        aspect=16/9,
        near=0.1,
        far=10000,
        X=1280,
        Y=720
    )
    return camera

def _getLight():
    return jnp.array([[10, 0, 0, 1, 1, 1, 0]], float)


def _build_scene(m_model : mjx.Model):
    scene = Scene.create(_getCamera(), _getLight(), 1, 1)

    def perGeom(geom_type, geom_rbga, geom_size, geom_pos, scene : Scene):
        transform = jnp.identity(4, float).at[3, :3].set(-geom_pos)
        print(transform)
        if geom_type == 6: #Box
            model = create_cube(geom_size[0], jnp.array([[[geom_rbga[:3]]]]), jnp.array([[[[0.05, 0.05, 0.05]]]]), transform)
        elif geom_type == 2: #Sphere
            model = create_capsule(geom_size[0], 0, 1, jnp.array([[[geom_rbga[:3]]]]), jnp.array([[[[0.05, 0.05, 0.05]]]]), transform)
        _, scene = Scene.addModel(scene, model)
        return scene
    
    
    for i in range(m_model.geom_rgba.shape[0]):
        scene = perGeom(m_model.geom_type[i], m_model.geom_rgba[i], m_model.geom_size[i], m_model.geom_pos[i], scene)
    
    return scene


def _updateScene(scene : Scene, m_data : mjx.Data):
    def perGeom(scene : Scene, idx, pos, rot):
        transform = jnp.identity(4, float).at[3, :3].set(-pos)
        rot = jnp.identity(4, float).at[0, :3].set([-rot[0], -rot[1], -rot[2]]).at[1, :3].set([-rot[3], -rot[4], -rot[5]]).at[2,:3].set([-rot[6], -rot[7], -rot[8]])
        transform = transform @ rot
        return Scene.transformModel(scene, idx, transform)
    
    for i in range(m_data.geom_xpos.shape[0]):
        scene = perGeom(scene, i, m_data.geom_xpos[i], m_data.geom_xmat[i])
    
    return scene
    
    
Render.loadVertexShaders(stdVertexShader, stdVertexExtractor)
Render.loadFragmentShaders(stdFragmentShader, stdFragmentExtractor)
jit_step = jax.jit(mjx.step)
duration = 3.8  # (seconds)
framerate = 30  # (Hz)

myscene = _build_scene(mjx_model)
Render.add_Scene(myscene, "Test")

with jax.profiler.trace("jax-trace-brax"):
    frames = []
    #pixels = Render.render_forward()
    #frames.append(pixels.astype("uint8"))
    mujoco.mj_resetData(mj_model, mj_data)
    mjx_data = mjx.put_data(mj_model, mj_data)
    while mjx_data.time < duration:
        mjx_data = jit_step(mjx_model, mjx_data)
        if len(frames) < mjx_data.time * framerate:
            mj_data = mjx.get_data(mj_model, mjx_data)
            Render.scenes["Test"] = _updateScene(Render.scenes["Test"], mj_data)
            
            start = time.time_ns()
            pixels = jnp.transpose(Render.render_forward(), [1, 0, 2])
            end = time.time_ns()
            print(f"Time to render: {(end - start) / 1000 / 1000}ms")
            frames.append(pixels.astype("uint8"))


#import matplotlib.pyplot as plt
#filenames = []

#plt.imshow(jnp.transpose(frames[3], [1, 0, 2]))
#for i, frame in enumerate(frames):
    #plt.imshow(jnp.transpose(frame, [1, 0, 2]))
    #plt.savefig(f"./brax_output/output{i}.png")
    #filenames.append(f"./brax_output/output{i}.png")


import imageio
images = []
imageio.mimsave('./brax_output/output.gif', frames)

    