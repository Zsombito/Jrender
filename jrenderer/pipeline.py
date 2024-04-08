from .scene import Scene
from .object import Model
from .r_types import Float, Integer, BoolV, Position, Face, PosXNorm, Vec3f, Matrix4, Normal, UV
from jax import vmap, jit
import jax
import jax.numpy as jnp
from typing import Callable, List, Tuple, Any
from .util_functions import homogenousToCartesian
import numpy as np


class Render:
    #Scene variables
    scenes : dict[str, Scene] = {}
    rendered_scenes : list[str] = []
    scene_update : bool = True
    clipping_batch = 4
    vertexExtractor : Callable = None
    vertexShader : Callable = None

    
    #Vertex variables 
    
    #Reset pipeline
    @staticmethod
    def flush_pipeline():
        Render.scenes.clear()
        Render.rendered_scenes.clear()
        Render.scene_update = True
    
    #Handle scenes
    @staticmethod
    def add_Scene(scene : Scene, name : str, addToRender: bool = True):
        if name in Render.scenes:
            raise Exception(f"A scene with name '{name}' already exists!")
        
        Render.scenes[name] = scene
        if addToRender:
            Render.rendered_scenes.append(name)

        Render.scene_update = True

    @staticmethod      
    def remove_Scene(name : str):
        if name not in Render.scenes:
            raise Exception(f"The scene with name '{name}' does not exist")

        if name in Render.rendered_scenes:
            Render.rendered_scenes.pop(name)
        
        Render.rendered_scenes.pop(name)
        Render.scene_update = True
        
    @staticmethod    
    def add_Scene_to_render(name : str):
        if name not in Render.scenes:
            raise Exception(f"The scene with name '{name}' does not exist")

        if name not in Render.rendered_scenes:
            Render.rendered_scenes.add(name)

        Render.scene_update = True

    
    @staticmethod
    def remove_Scene_from_render(name : str):
        if name  in Render.rendered_scenes:
            Render.rendered_scenes.pop(name)

        Render.scene_update = True
    
    @staticmethod
    def callSceneUpdate():
        Render.scene_update = True
    
    @staticmethod
    def loadVertexShaders(vertexShader : Callable, vertexExtractor : Callable):
        Render.vertexExtractor = vertexExtractor
        Render.vertexShader = vertexShader


        
    @jit
    def _applyVertexShader():
        result = []
        for scene in Render.rendered_scenes:
            #Extract vertex information
            args, argAxis, extra = Render.vertexExtractor(Render.scenes[scene])
            shaded_args,shaded_extra = vmap(Render.vertexShader, argAxis)(*args)
            result.append((shaded_args, shaded_extra, extra))
        return result

        
    @jit
    def _clipVertex(position : Position, near : float, far : float, top : Vec3f, bot : Vec3f, left : Vec3f, right : Vec3f) :
        pos3D = homogenousToCartesian(position)
        return ((pos3D[2] > near) & (pos3D[2] < far) & (jnp.dot(pos3D, top) < 0) & (jnp.dot(pos3D, bot) < 0) & (jnp.dot(pos3D, left) < 0) & (jnp.dot(pos3D, right) < 0))
    
    @jit
    def getFrustrumParams(near : float, far: float, fov: float, aspect: float):
        hw = jnp.tan(jnp.pi * fov/360) * near
        hh = hw * (1 / aspect)
        nw = jnp.array([-hw, hh, near])
        ne = jnp.array([hw, hh, near])
        se = jnp.array([hw, -hh, near])
        sw = jnp.array([-hw, -hh, near])
        top = jnp.cross(nw, ne)
        right = jnp.cross(ne, se)
        bot = jnp.cross(se, sw)
        left = jnp.cross(sw, nw)
        return (near, far, top, bot, left, right)

        
    @jit
    def filterFaces(face : Face, mask : BoolV):
        return mask[face[0]] + mask[face[1]] + mask[face[2]] 

    @jit
    def clip(position : Float[Position, "idx"], face : Integer[Face, "idx"], fov : float, aspect : float):
        cameraArgs = Render.getFrustrumParams(0.01, 1.0, fov, aspect)
        mask = vmap(Render._clipVertex, [0, None, None, None, None, None, None])(position, *cameraArgs)
        return vmap(Render.filterFaces, [0, None])(face, mask)

    @jit
    def viewPort(position : Float[Position, "idx"], viewPort: Matrix4):
        position = position @ viewPort
        pos3 = jnp.apply_along_axis(homogenousToCartesian, 1, position)
        return pos3

    
    
    def geometryStage():
        with jax.named_scope("Vertex shading"):
            vertexShaded : List[Tuple[PosXNorm, Any, Any]] = Render._applyVertexShader()
        for i, perVertex in enumerate(vertexShaded):
            (pos, norm), save, (modelIds, face) = perVertex
            #rem = pos.shape[0] % Render.clipping_batch 
            #if rem != 0:
                #batch = pos.shape[0] // Render.clipping_batch +  1
                #posDumy = Render.clipping_batch - rem
                #posAug = jnp.concatenate((pos, jnp.zeros((posDumy, 4), float)), axis=0)
            #else:
                #batch = pos.shape[0] // Render.clipping_batch
                #posAug = pos
            #posAug = posAug.reshape(batch, Render.clipping_batch,4)


            #Clipping
            with jax.named_scope("Clipping"):
                camera = Render.scenes[Render.rendered_scenes[i]].camera
                face_mask = Render.clip(pos, face, camera.fov, camera.aspect)
                face = face[face_mask, :]
            
            #Viewport transform
            with jax.named_scope("Viewport transform"):
                pos3 = Render.viewPort(pos, camera.viewPortMatrix)

            #Create corners
            with jax.named_scope("Rasterizing"): 
                corners = vmap(Render.getCorners, [0, None])(face, pos3)
                gridX, gridY = jnp.arange(0, camera.X), jnp.arange(0, camera.Y)
                fragments = vmap(Render.mapX, [0, None, None])(gridX, gridY, corners) #Lines of fragment value
            

            #Create fragements
            #fragments, mask, lerp = vmap(Render.getFragments, [0, 0, 0, 0, 0])(*minmax, corners)
            return pos3, corners

    @jit
    def primitiveHit(frag, corners):
        v0 = corners[1, :2] - corners[0, :2] # b-a
        v1 = corners[2, :2] - corners[0, :2] # c-a
        v2 = frag - corners[0, :2] # p-a

        d00 = jnp.dot(v0, v0)
        d01 = jnp.dot(v0, v1)
        d11 = jnp.dot(v1, v1)
        d20 = jnp.dot(v2, v0)
        d21 = jnp.dot(v2, v1)
        denom = d00*d11 - d01 * d01

        alpha = (d11 * d20 - d01 * d21) / denom
        beta = (d00 * d21 - d01 * d20) / denom
        gamma = 1.0 - alpha - beta

        return ((alpha >= 0) & (beta >= 0) & (gamma >= 0), jnp.array([alpha, beta, gamma]))

    
    @jit
    def getCorners(face : Face, position : Float[Vec3f, "idx"]):
        corners = jnp.array([position[face[0]], position[face[1]], position[face[2]]])
        return corners

    @jit
    def mapY(x, y, corners):
        return vmap(Render.primitiveHit, (None, 0))(jnp.array([x,y]), corners)

    @jit
    def mapX(x, y, corners):
        return vmap(Render.mapY, [None, 0, None])(x, y, corners)
    
        
        
        
    
