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
from jrenderer.camera_linker import CameraLink
from jrenderer.lights import Light
from jrenderer.capsule import create_capsule
from jrenderer.cube import create_cube
from jrenderer.plane import create_plane
from jrenderer.pipeline_brax_raster_optimized import Render
from jrenderer.model import Model
from jrenderer.scene import Scene
from jrenderer.shader import stdVertexExtractor, stdVertexShader, stdFragmentExtractor, stdFragmentShader
import mujoco
from mujoco import mjx
import mediapy as media
from jaxtyping import Float, Array, Integer

import mujoco.testdata

import time



class BraxRenderer(NamedTuple):
    scene : Scene
    cameras : list[Camera]
    camera_links : list[CameraLink]
    geom_offset : Float[Array, "idx 3"]
    geom_rotation : Float[Array, "idx 4"]
    geom_link_idx : Integer[Array, "idx"]

    @staticmethod
    def _getLight():
        return jnp.array([[0, 0, 13000, 1, 1, 1, 0]], float)

    @staticmethod
    def _getCamera():
        camera = Camera.create(
            position=jnp.array([7, 7, 7.0]) ,
            target=jnp.zeros(3),
            up=jnp.array([0, 0.0, 1.0]),
            fov=75,
            aspect=1080/720,
            near=0.1,
            far=10000,
            X=1080,
            Y=720
        )
        return camera


    @staticmethod
    def _extractGeoms(sys : base.System):
        scene : Scene = Scene.create(BraxRenderer._getLight(), 1, 2)
        geom_offset = jnp.empty((0, 3), float)
        geom_rotation = jnp.empty((0,4), float)
        geom_link_idx = jnp.empty(0, int)

        
        for geom_idx in range(sys.ngeom):
            geom_type = sys.geom_type[geom_idx]
            defaultDiffText = jnp.array([[[sys.geom_rgba[geom_idx][:3],sys.geom_rgba[geom_idx][:3]]]])
            defaultSpecText = jnp.array([[[[0.05, 0.05, 0.05], [0.05, 0.05, 0.05]]]])
            if geom_type == 6: #Box
                model = create_cube(sys.geom_size[geom_idx][0], defaultDiffText, defaultSpecText)
            elif geom_type == 2: #Sphere
                model = create_capsule(sys.geom_size[geom_idx][0], 0, 2, defaultDiffText, defaultSpecText)
            elif geom_type == 3: #Capsule
                if sys.geom_size[geom_idx].shape[0] == 1:
                    model = create_capsule(sys.geom_size[geom_idx][0], 1 * sys.geom_size[geom_idx][0], 2, defaultDiffText, defaultSpecText)
                else:
                    model = create_capsule(sys.geom_size[geom_idx][0], sys.geom_size[geom_idx][1], 2, defaultDiffText, defaultSpecText)
            #elif geom_type == 0: #Plane
                #print("HERE")
                #model = create_plane(sys.geom_size[geom_idx], jnp.array([[[[0.5, 0.5, 0.5], [0.75, 0.75, 0.75]]]]), defaultSpecText)
                

            else:
                continue

            _, scene = scene.addModel(model)
            geom_link_idx = jnp.append(geom_link_idx, sys.geom_bodyid[geom_idx] - 1)
            geom_offset = jnp.append(geom_offset, jnp.array([sys.geom_pos[geom_idx]]), 0)
            geom_rotation = jnp.append(geom_rotation, jnp.array([sys.geom_quat[geom_idx]]), 0)
        
        return scene, geom_offset, geom_rotation, geom_link_idx


    @staticmethod
    def create(sys : base.System):
        scene, geom_offset, geom_rotation, geom_link_idx = BraxRenderer._extractGeoms(sys)
        camera = BraxRenderer._getCamera()
        cameraLinker = CameraLink(0, 0)
        return BraxRenderer(scene, [camera], [cameraLinker], geom_offset, geom_rotation, geom_link_idx)

    @jax.jit
    def _perGeomUpdate(geom_off : Float[Array, "3"], geom_rot, geom_link_idx, xpos, xrot):
        pos = jnp.where(geom_link_idx != -1, xpos[geom_link_idx],  jnp.zeros(3, float))
        rot = jnp.where(geom_link_idx != -1, xrot[geom_link_idx], jnp.array([1, 0, 0, 0], float))
        new_off = pos + math.rotate(geom_off, rot)
        new_rot = math.quat_mul(rot, geom_rot)
        transition_matrix = jnp.identity(4, float).at[3, :3].set(new_off)
        rotation_matrix = jnp.identity(4,float).at[:3, :3].set(jnp.transpose(math.quat_to_3x3(new_rot)))
        return rotation_matrix @ transition_matrix

    @jax.jit
    def renderState(self, state : brax.State):
        new_mdl_matricies = jax.vmap(BraxRenderer._perGeomUpdate, [0, 0, 0, None, None])(self.geom_offset, self.geom_rotation, self.geom_link_idx, state.x.pos, state.x.rot)
        scene = self.scene._replace(mdlMatricies=new_mdl_matricies)
        camera = self.camera_links[0].updateCamera(self.cameras[0], state.x.pos, state.x.rot, self.geom_offset, self.geom_rotation, self.geom_link_idx)
        return jnp.transpose(Render.render_forward(scene, camera), [1,0,2]).astype("uint8")

        
    def changeCameraLinker(self, mode : int, target : int):
        link = CameraLink(mode, target)
        return self._replace(camera_links = [link])

        

    