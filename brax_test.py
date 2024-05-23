
from typing import Iterable, NamedTuple, Optional

import jax
from jax import numpy as jp
import numpy as onp

import brax
from brax import math

import trimesh

from jrenderer.camera import Camera
from jrenderer.lights import Light
from jrenderer.capsule import create_capsule
from jrenderer.cube import create_cube
from jrenderer.pipeline import Render
from jrenderer.model import Model
from jrenderer.scene import Scene


from typing import Iterable, NamedTuple, Optional

import jax
from jax import numpy as jnp
import numpy as np
from jaxtyping import Float, Integer, Array

import brax
from brax import base, envs, math, positional
from brax.envs.base import Env, PipelineEnv, State
from brax.envs import ant
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model 


import trimesh

from jrenderer.camera import Camera
from jrenderer.lights import Light
from jrenderer.capsule import create_capsule
from jrenderer.cube import create_cube
from jrenderer.pipeline_brax_without_clipping import Render
from jrenderer.model import Model
from jrenderer.scene import Scene
from jrenderer.shader import stdVertexExtractor, stdVertexShader, stdFragmentExtractor, stdFragmentShader
import mujoco
from mujoco import mjx
import mediapy as media

import mujoco.testdata

import time

class GeomOffset(NamedTuple):
    rot : Float[Array, "4"]
    off : Float[Array, "3"]
    link_idx : int
    model_idx : int

    def _transform(self, rot : Float[Array, "4"], pos : Float[Array, "3"]):
        new_off = pos + math.rotate(self.off, rot)
        new_rot = math.quat_mul(rot, self.rot) 
        return GeomOffset(new_rot, new_off, self.link_idx, self.model_idx)


def _calculateBodyOffset(sys : brax.System, body_idx : int):
    if sys.body_parentid[body_idx] == 0:
        return jnp.zeros(3, float), jnp.zeros(4, float).at[0].set(1.)
    else:
        parent_off, parent_rot =  _calculateBodyOffset(sys, sys.body_parentid[body_idx])
        off = sys.body_pos[body_idx] + parent_off
        rot = math.quat_mul(parent_rot, sys.body_quat[body_idx])
        return  off, rot


def buildScene(sys : brax.System) -> tuple[Scene, list[GeomOffset]]:
    scene = Scene.create(_getLight(), 1, 1)
    geom_offsets : list[GeomOffset] = []
    for geom_idx in range(sys.ngeom):
        geom_type = sys.geom_type[geom_idx]
        if geom_type == 6: #Box
            model = create_cube(sys.geom_size[geom_idx][0], jnp.array([[[sys.geom_rgba[geom_idx][:3]]]]), jnp.array([[[[0.05, 0.05, 0.05]]]]))
        elif geom_type == 2: #Sphere
            model = create_capsule(sys.geom_size[geom_idx][0], 0, 2, jnp.array([[[sys.geom_rgba[geom_idx][:3]]]]), jnp.array([[[[0.05, 0.05, 0.05]]]]))
        elif geom_type == 3: #Capsule
            if sys.geom_size[geom_idx].shape[0] == 1:
                model = create_capsule(sys.geom_size[geom_idx][0], 1 * sys.geom_size[geom_idx][0], 2, jnp.array([[[sys.geom_rgba[geom_idx][:3]]]]), jnp.array([[[[0.05, 0.05, 0.05]]]]))
            else:
                model = create_capsule(sys.geom_size[geom_idx][0], sys.geom_size[geom_idx][1], 2, jnp.array([[[sys.geom_rgba[geom_idx][:3]]]]), jnp.array([[[[0.05, 0.05, 0.05]]]]))
                
        else:
            continue
        
        model_idx, scene = scene.addModel(model)
        geom_off = sys.geom_pos[geom_idx]
        geom_rot = sys.geom_quat[geom_idx]
        geom_link_idx = sys.geom_bodyid[geom_idx] - 1
        geom_offsets.append(GeomOffset(geom_rot, geom_off, geom_link_idx, model_idx))

    return scene, geom_offsets

def applyGeomOffsets(scene : Scene, geom_offsets : list[GeomOffset]) -> Scene:
    for geom_offset in geom_offsets:
        transition_matrix = jnp.identity(4, float).at[3, :3].set(geom_offset.off)
        rotation_matrix = jnp.identity(4,float).at[:3, :3].set(jnp.transpose(math.quat_to_3x3(geom_offset.rot)))
        scene = scene.transformModel(geom_offset.model_idx, rotation_matrix @ transition_matrix)
    
    return scene


def _getCamera():
    camera = Camera.create(
        position=jnp.array([10, 10, 10.0]) ,
        target=jnp.zeros(3),
        up=jnp.array([0, 0.0, 1.0]),
        fov=75,
        aspect=16/9,
        near=0.1,
        far=10000,
        X=1280//2,
        Y=720//2
    )
    return camera

def _getLight():
    return jnp.array([[0, 0, 13000, 1, 1, 1, 0]], float)


def updateGeomData(geom_datas : list[GeomOffset], state : brax.State):
    new_geom_data = []
    for i, geom_data in enumerate(geom_datas):
        if geom_data.link_idx != -1:
            d_pos = state.x.pos[geom_data.link_idx]
            d_rot = state.x.rot[geom_data.link_idx]
            new_geom_data.append(geom_data._transform(d_rot, d_pos))
    
    return new_geom_data



Render.loadVertexShaders(stdVertexShader, stdVertexExtractor)
Render.loadFragmentShaders(stdFragmentShader, stdFragmentExtractor)

human = ant.Ant()

scene, geom_offsets = buildScene(human.sys)
scene = applyGeomOffsets(scene, geom_offsets)
camera = _getCamera()

import pickle

with open('states_ant.pkl', 'rb') as f:
    states : list[brax.State] = pickle.load(f)
print(states[0].pipeline_state.x.pos[3])
print(states[80].pipeline_state.x.pos[3])

frames = []

for i in range(40):
    curr_geom_offsets = updateGeomData(geom_offsets, states[i].pipeline_state)
    scene = applyGeomOffsets(scene, curr_geom_offsets)
    start = time.time_ns()
    pixels = jax.block_until_ready(jnp.transpose(Render.render_forward(scene, camera), [1, 0, 2]))
    end = time.time_ns()
    print(f"Frame{i} took: {(end - start) / 1000 / 1000} ms")
    pixels = pixels.astype("uint8")
    frames.append(pixels)

print("Making giff")
import imageio
imageio.mimsave('./brax_output/output.gif', frames)