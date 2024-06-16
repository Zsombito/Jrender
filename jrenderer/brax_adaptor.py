from typing import NamedTuple, Any

import jax
from jax import numpy as jnp
from jaxtyping import Float, Integer, Array

import brax
from brax import base, math


from jrenderer.camera import Camera
from jrenderer.camera_linker import CameraLink
from jrenderer.capsule import create_capsule
from jrenderer.cube import create_cube
from jrenderer.plane import create_plane
from jrenderer.pipeline_brax_without_clipping import Render
from jrenderer.scene import Scene
from jaxtyping import Float, Array, Integer




class BraxRenderer(NamedTuple):
    scene : Scene
    camera : Camera
    camera_link : CameraLink
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
            elif geom_type == 0: #Plane
                model = create_plane(sys.geom_size[geom_idx], jnp.array([[[[0.5, 0.5, 0.5], [0.75, 0.75, 0.75]]]]), defaultSpecText)
                

            else:
                continue

            _, scene = scene.addModel(model)
            geom_link_idx = jnp.append(geom_link_idx, sys.geom_bodyid[geom_idx] - 1)
            geom_offset = jnp.append(geom_offset, jnp.array([sys.geom_pos[geom_idx]]), 0)
            geom_rotation = jnp.append(geom_rotation, jnp.array([sys.geom_quat[geom_idx]]), 0)
        
        return scene, geom_offset, geom_rotation, geom_link_idx


    @staticmethod
    def create(sys : base.System):
        """
        Creates a BraxRenderer instance containing the scene and camera from a brax system
        """
        scene, geom_offset, geom_rotation, geom_link_idx = BraxRenderer._extractGeoms(sys)
        camera = BraxRenderer._getCamera()
        cameraLinker = CameraLink(0, 0)
        return BraxRenderer(scene, camera, cameraLinker, geom_offset, geom_rotation, geom_link_idx)

    @jax.jit
    def _perGeomUpdate(geom_off : Float[Array, "3"], geom_rot, geom_link_idx, xpos, xrot):
        pos = jnp.where(geom_link_idx != -1, xpos[geom_link_idx],  jnp.zeros(3, float))
        rot = jnp.where(geom_link_idx != -1, xrot[geom_link_idx], jnp.array([1, 0, 0, 0], float))
        new_off = pos + math.rotate(geom_off, rot)
        new_rot = math.quat_mul(rot, geom_rot)
        transition_matrix = jnp.identity(4, float).at[3, :3].set(new_off)
        rotation_matrix = jnp.identity(4,float).at[:3, :3].set(jnp.transpose(math.quat_to_3x3(new_rot)))
        return rotation_matrix @ transition_matrix

    def _renderState_unjitted(self, state : brax.State, loop_unroll = 100):
        """
        Rendering the scene with the updated parameters based on state
        """
        new_mdl_matricies = jax.vmap(BraxRenderer._perGeomUpdate, [0, 0, 0, None, None])(self.geom_offset, self.geom_rotation, self.geom_link_idx, state.x.pos, state.x.rot)
        scene = self.scene._replace(mdlMatricies=new_mdl_matricies)
        camera = self.camera_link.updateCamera(self.camera, state.x.pos, state.x.rot, self.geom_offset, self.geom_rotation, self.geom_link_idx)
        return jnp.transpose(Render.render_forward(scene, camera, loop_unroll = loop_unroll), [1,0,2]).astype("uint8")
    
    renderState = jax.jit(_renderState_unjitted, static_argnames=["loop_unroll"])

        

        
    def config(self, in_config : dict[str, Any]):
        """
        Allows for configuration of the BraxRenderer's camera parameters
        Default valus:
        X: 1080
        Y: 720
        FoV: 90
        CamPos: [7, 7, 7]
        CamTarget: [0, 0 , 0]
        CamUp: [0,0,1]
        Near: 0.1
        Far: 10000
        CamLinkMode: 0 [1 for tracking]
        CamLinkTarget: 0
        """
        config = {
            "X":1080,
            "Y":720,
            "FoV": 75,
            "CamPos": jnp.ones(3, float) * 7,
            "CamTarget": jnp.zeros(3, float),
            "CamUp": jnp.array([0,0,1]),
            "Near": 0.1,
            "Far": 10000,
            "CamLinkMode" : 0,
            "CamLinkTarget": 0,
        }

        for key, value in in_config.items():
            config[key]=value
        
        camera = Camera.create(
            position=config["CamPos"],
            target=config["CamTarget"],
            up=config["CamUp"],
            fov=config["FoV"],
            aspect=config["X"]/config["Y"],
            near=config["Near"],
            far=config["Far"],
            X=config["X"],
            Y=config["Y"]
        )

        camera_link = CameraLink(config["CamLinkMode"], config["CamLinkTarget"])

        return self._replace(camera = camera, camera_link = camera_link)

    