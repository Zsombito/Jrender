from typing import Iterable, NamedTuple, Optional

import jax
from jax import numpy as jp
import numpy as onp

import brax
from brax import base, envs, math

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
from brax.envs import humanoid
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model 


import trimesh

from jrenderer.camera import Camera
from jrenderer.lights import Light
from jrenderer.capsule import create_capsule
from jrenderer.cube import create_cube
from jrenderer.pipeline import Render
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
        new_rot = math.quat_mul(self.rot, rot)
        return GeomOffset(new_rot, new_off, self.link_idx, self.model_idx)

def _convertBodyNames(body_adr : np.ndarray, names : bytes) -> list[str]:
    body_names : list[str] = []
    for adr in body_adr:
        end = adr
        while(names[end] != 0):
            end += 1
        body_names.append(names[adr:end].decode())
    return body_names

def _calculateBodyOffset(sys : brax.System, body_idx : int):
    if sys.body_parentid[body_idx] == 0:
        return jnp.zeros(3, float), jnp.zeros(4, float).at[0].set(1.)
    else:
        parent_off, parent_rot =  _calculateBodyOffset(sys, sys.body_parentid[body_idx])
        off = sys.body_pos[body_idx] + parent_off
        rot = math.quat_mul(parent_rot, sys.body_quat[body_idx])
        return  off, rot

def _addBodytoScene(scene : Scene, sys : brax.System, body_idx : int, link_idx : int) -> tuple[Scene, list[GeomOffset]]:
    body_off, body_rot = _calculateBodyOffset(sys, body_idx)
    extra_geom_datas : list[GeomOffset] = []
    for i in range(sys.body_geomnum[body_idx]):
        geom_idx = sys.body_geomadr[body_idx] + i


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

        geom_off = sys.geom_pos[geom_idx] + body_off
        geom_rot = math.quat_mul(body_rot, sys.geom_quat[geom_idx])
        print(f"Geom{i}: {geom_off} off, {body_rot} -> {geom_rot} rot")
        extra_geom_datas.append(GeomOffset(geom_rot, geom_off, link_idx, model_idx))

    return scene, extra_geom_datas


def buildScene(sys : brax.System) -> tuple[Scene, list[GeomOffset]]:
    scene = Scene.create(_getCamera(), _getLight(), 1, 1)
    body_names = _convertBodyNames(sys.name_bodyadr, sys.names)
    geom_offsets : list[GeomOffset] = []
    for i in range(sys.nbody):
        link_idx = -1
        for j, link_name in enumerate(sys.link_names):
            if link_name == body_names[i]:
                link_idx = j
                break
        
        print("---------------------")
        print(f"{body_names[i]}")
        print(f"Body offset: {_calculateBodyOffset(sys, i)}")
        print("Geoms:")
        scene, new_geom_offsets = _addBodytoScene(scene, sys, i, link_idx)
        print("--------------------------------------")
        geom_offsets = geom_offsets + new_geom_offsets
    
    return scene, geom_offsets

def applyGeomOffsets(scene : Scene, geom_offsets : list[GeomOffset]) -> Scene:
    for geom_offset in geom_offsets:
        transition_matrix = jnp.identity(4, float).at[3, :3].set(geom_offset.off)
        rotation_matrix = jnp.identity(4,float).at[:3, :3].set(jnp.transpose(math.quat_to_3x3(geom_offset.rot)))
        scene = scene.transformModel(geom_offset.model_idx, rotation_matrix @ transition_matrix)
    
    return scene


def _getCamera():
    camera = Camera.create(
        position=jnp.array([3, 0, 0.0]) ,
        target=jnp.zeros(3),
        up=jnp.array([0, 0.0, 1.0]),
        fov=75,
        aspect=16/9,
        near=0.1,
        far=10000,
        X=1280,
        Y=720
    )
    return camera

def _getLight():
    return jnp.array([[0, 0, 13000, 1, 1, 1, 0]], float)


def _build_scene(m_model : brax.System):
    scene = Scene.create(_getCamera(), _getLight(), 1, 1)

    def perGeom(geom_type, geom_rbga, geom_size, geom_pos, geom_quat, scene : Scene):
        transform = jnp.identity(4, float).at[3, :3].set(-geom_pos)
        print(geom_type)
        print(geom_quat)
        if geom_type == 6: #Box
            model = create_cube(geom_size[0], jnp.array([[[[0, 1, 0]]]]), jnp.array([[[[0.05, 0.05, 0.05]]]]), transform)
        elif geom_type == 2: #Sphere
            model = create_capsule(geom_size[0], 0, 1, jnp.array([[[[1., 0.0, 0.0]]]]), jnp.array([[[[0.05, 0.05, 0.05]]]]), transform)
        elif geom_type == 3: #Capusule
            print(geom_size[1]/2)
            model = create_capsule(geom_size[0], geom_size[1], 2, jnp.array([[[[1,1,1]]]]), jnp.array([[[[0.05, 0.05, 0.05]]]]), transform)
        else:
            return scene
        _, scene = Scene.addModel(scene, model)
        return scene
    
    
    for i in range(m_model.geom_rgba.shape[0]):
        print(m_model.geom_bodyid[i])
        scene = perGeom(m_model.geom_type[i], m_model.geom_rgba[i], m_model.geom_size[i], m_model.geom_pos[i], m_model.geom_quat[i],scene)
    
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

human = humanoid.Humanoid()

#scene = _build_scene(human.sys)
#Render.add_Scene(scene, "Test")
#pixels = jnp.transpose(Render.render_forward(), [1, 0, 2])
#end = time.time_ns()
#pixels = pixels.astype("uint8")

#import matplotlib.pyplot as plt

#plt.imshow(pixels)
#plt.savefig('humanoid.png')  # pyright: ignore[reportUnknownMemberType]

#print(human.sys.link_names)
#print(human.sys.ngeom)
#print(human.sys.nbody)
#print(human.sys.link_types)
#print(human.sys.geom_bodyid)
#print(human.sys.body_pos)
#print(human.sys.geom_pos)
#print(human.sys.body_parentid)
#print(human.sys.link_parents)
#print(human.sys.body_geomnum)
#print(human.sys.njnt)
#print(human.sys.body_geomadr)
#human.step

scene, geom_offsets = buildScene(human.sys)
scene = applyGeomOffsets(scene, geom_offsets)


Render.add_Scene(scene, "Test")
pixels = jnp.transpose(Render.render_forward(), [1, 0, 2])
pixels = pixels.astype("uint8")

import matplotlib.pyplot as plt

plt.imshow(pixels)
plt.savefig('humanoid.png')  # pyright: ignore[reportUnknownMemberType]