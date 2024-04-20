import jax.experimental
import jax.experimental.host_callback
from .scene import Scene
from .object import Model
from .r_types import Float, Integer, BoolV, Position, Face, PosXNorm, Vec3f, Matrix4, Normal, UV
from jax import vmap, jit
import jax
import jax.numpy as jnp
import jax.lax as lax
from typing import Callable, List, Tuple, Any
from .util_functions import homogenousToCartesian
import numpy as np
from functools import partial


class Render:
    #Scene variables
    scenes : dict[str, Scene] = {}
    rendered_scenes : list[str] = []
    scene_update : bool = True
    clipping_batch = 4
    currentScene : Scene = None
    rasterGrid = 50
    faceBatch = 5000
    fragBatch = 10000

    
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

###################################################################
#                  VertexShading & Cliping                        #
###################################################################

        
    @jit
    def _applyVertexShader():
        #Extract vertex information
        args, argAxis, face, perVertexExtra = Render.currentScene.vertexExtractor(Render.currentScene)
        (pos, norm), shaded_PerVertexExtra = vmap(Render.currentScene.vertexShader, argAxis)(*args)
        return ((pos, norm, face), perVertexExtra, shaded_PerVertexExtra)

        

    @jit
    def __clipVertex(position : Position, near : float, far : float, top : Vec3f, bot : Vec3f, left : Vec3f, right : Vec3f) :
        pos3D = homogenousToCartesian(position)
        return ((pos3D[2] > near) & (pos3D[2] < far) & (jnp.dot(pos3D, top) < 0) & (jnp.dot(pos3D, bot) < 0) & (jnp.dot(pos3D, left) < 0) & (jnp.dot(pos3D, right) < 0))
    
    @jit
    def __getFrustrumParams(near : float, far: float, fov: float, aspect: float):
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
    def __filterFaces(face : Face, mask : BoolV):
        return mask[face[0]] + mask[face[1]] + mask[face[2]] 

    @jit
    def _clip(position : Float[Position, "idx"], face : Integer[Face, "idx"], fov : float, aspect : float):
        cameraArgs = Render.__getFrustrumParams(0.01, 1.0, fov, aspect)
        mask = vmap(Render.__clipVertex, [0, None, None, None, None, None, None])(position, *cameraArgs)
        return vmap(Render.__filterFaces, [0, None])(face, mask)

    @jit
    def _viewPort(position : Float[Position, "idx"], viewPort: Matrix4):
        position = position @ viewPort
        pos3 = jnp.apply_along_axis(homogenousToCartesian, 1, position)
        return pos3

    
    @jit
    def geometryStage():
        camera = Render.currentScene.camera 

        with jax.named_scope("Vertex shading"):
            (pos, norm, face), perVertexExtra, shaded_perVertexExtra  = Render._applyVertexShader()
        
        with jax.named_scope("Clipping"):
            face_mask = Render._clip(pos, face, camera.fov, camera.aspect)
        
        with jax.named_scope("Viewport transform"):
            pos3 = Render._viewPort(pos, camera.viewPortMatrix)
        
        return face_mask, (pos3, norm, face), perVertexExtra, shaded_perVertexExtra




###################################################################
#                     Fragment Generation(A)                      #
###################################################################
    @jit
    def _getCorners(face : Face, position : Float[Vec3f, "idx"]):
        corners = jnp.array([position[face[0]], position[face[1]], position[face[2]]])
        return corners

    @jit
    def ___primitiveHit(frag, corners):
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
    def __mapY(x, y, corners):
        return vmap(Render.___primitiveHit, (None, 0))(jnp.array([x,y]), corners)

    @jit
    def _mapX(x, y, corners):
        return vmap(Render.__mapY, [None, 0, None])(x, y, corners)
    
    @jit
    def generateFragments(pos3, face):
        camera = Render.currentScene.camera
        with jax.named_scope("Creating primitives"): 
            corners = vmap(Render._getCorners, [0, None])(face, pos3)

        with jax.named_scope("Creating fragments"):
            gridX, gridY = jnp.arange(0, camera.X), jnp.arange(0, camera.Y)
            fragments = vmap(Render._mapX, [0, None, None])(gridX, gridY, corners) 
        
        return fragments


###################################################################
#                     Fragment Generation(C)                      #
###################################################################


    @jit
    def _getMinMax(corners):
        minX = corners[:, 0].min()
        maxX = corners[:, 0].max()
        minY = corners[:, 1].min()
        maxY = corners[:, 1].max()
        return jnp.array([minX, maxX, minY, maxY], int)

    @jit
    def __filterBracket(minmax, bMinX, bMaxX, bMinY, bMaxY):
        return (minmax[0] <= bMaxX) & (bMinX <= minmax[1]) & (minmax[2] <= bMaxY) & (bMinY <= minmax[3])

    @jit
    def _checkBrackets(minmaxs, corners):

        def map_Y(y, bMinX, bMaxX): 
            return vmap(Render.__filterBracket, [0, None, None, None, None])(minmaxs, bMinX, bMaxX, y * Render.rasterGrid, (y + 1) * Render.rasterGrid)

        def map_X(x, brackY):
            return vmap(map_Y, [0, None, None])(brackY, x * Render.rasterGrid, (x+1) * Render.rasterGrid)
        
        def iota_Prim(x, y, idx):
            return jnp.array([x * Render.rasterGrid, y * Render.rasterGrid, idx])

        def iota_Y(x, y, corners):
            return vmap(iota_Prim, [None, None, 0])(x, y, lax.iota(int, corners.shape[0]))

        def iota_X(x, brackY, corners):
            return vmap(iota_Y, [None, 0, None])(x, brackY, corners)



        camera = Render.currentScene.camera
        bracketX = camera.X // Render.rasterGrid + 1
        bracketY = camera.Y // Render.rasterGrid + 1
        brackX = lax.iota(int, bracketX)
        brackY = lax.iota(int, bracketY)
        return vmap(iota_X, [0, None, None])(brackX, brackY, corners), vmap(map_X, [0, None])(brackX, brackY)

        




    @jit
    def generateBrackets(pos3, face):
        with jax.named_scope("Creating primitives"): 
            corners = vmap(Render._getCorners, [0, None])(face, pos3)
            minmax = vmap(Render._getMinMax, [0])(corners)
            brackets, bracket_mask = Render._checkBrackets(minmax, corners)
            return corners, brackets, bracket_mask




    
    @jit
    def _PerBracketPerPrimitive(bracket, corners):

        def map_X(x, gridY, corner):
            return vmap(Render.____primitiveHit_B, [None, 0, None])(x, gridY, corner)

        x, y, idx = bracket[0], bracket[1], bracket[2]
        gridX = lax.iota(int, Render.rasterGrid) + (jnp.ones(Render.rasterGrid, int) * x)
        gridY = lax.iota(int, Render.rasterGrid) + (jnp.ones(Render.rasterGrid, int) * y)
        corner = corners[idx]
        return vmap(map_X, [0, None, None])(gridX, gridY, corner)



    @jit
    def generateFragmentsFromBrackets(brackets, corners):
        return vmap(Render._PerBracketPerPrimitive, [0, None])(brackets, corners)
        
        

#####################################################################
#                     Main Pipeline Control                         #
#####################################################################
    @staticmethod
    def render():
        for scene in Render.rendered_scenes:
            Render.currentScene = Render.scenes[scene]
            face_mask, (pos3, norm, face), perVertexExtra, shaded_perVertexExtra = Render.geometryStage()
            face = face[face_mask, :]
            #Rearanging

            #frags = Render.generateFragments(pos3, face)

            corners, brackets, bracket_mask = Render.generateBrackets(pos3, face)
            brackets = brackets[bracket_mask]
            ##Rearanging

            frags = Render.generateFragmentsFromBrackets(brackets, corners)
            #print(frags)

        pass

    
        
        
        
    

###################################################################
#                     Fragment Generation(B)                      #
###################################################################


    
    def _bracket(corners, minmax, bracket):
        pass

    @jit
    def ____primitiveHit_B(x, y, corners):
        frag = jnp.array([x,y])
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

        return jnp.array([(alpha >= 0) & (beta >= 0) & (gamma >= 0), alpha, gamma, beta], float)
    
#===========Ture branch
    @jit
    def ___generateGrid(corner, gridX, gridY):
        def ____mapX_B(x, y, corner):
            return vmap(Render.____primitiveHit_B, [None, 0, None])(x, y, corner)

        return vmap(____mapX_B, [0, None, None])(gridX, gridY, corner)

#===========False branch

    @jit
    def ____filteredPrimitive(x,y):
        return jnp.array([False, 0, 0, 0], float)
    
    @jit
    def ___generateFilteredGrid(corner, gridX, gridY):
        def ____mapX_B(x, y):
            return vmap(Render.____filteredPrimitive, [None, 0])(x, y)

        return vmap(____mapX_B, [0, None])(gridX, gridY)
#============Branching
    @jit
    def __gridPerPrimitive(corner, minmax, bracket, gridX, gridY):
        condition = (minmax[0] <= bracket[1]) & (bracket[0] <= minmax[1]) & (minmax[2] <= bracket[3]) & (bracket[2] <= minmax[3])

        brack = lax.cond(condition, Render.___generateGrid, Render.___generateFilteredGrid, corner, gridX, gridY)
        return brack.reshape(50, 50, 1, 4)
    
    @jit
    def _createBracketing(corners, minmax):
        camera = Render.currentScene.camera
        bracketX = camera.X // Render.rasterGrid + 1
        bracketY = camera.Y // Render.rasterGrid + 1

        gridFalse = jnp.zeros((bracketX * Render.rasterGrid, bracketY * Render.rasterGrid, corners.shape[0], 1), bool)
        gridZeros = jnp.zeros((bracketX * Render.rasterGrid, bracketY * Render.rasterGrid, corners.shape[0], 3), float)
        grid = jnp.concatenate((gridFalse, gridZeros), axis=3)


        def inside_primitive_loop(x, y, i, grid):
            corner = corners[i]
            mm = minmax[i]
            gridX = jnp.arange(0, Render.rasterGrid) + (jnp.ones(Render.rasterGrid, int) * x)
            gridY = jnp.arange(0, Render.rasterGrid) + (jnp.ones(Render.rasterGrid, int) * y)
            bracket = jnp.array([x, x + Render.rasterGrid, y, y + Render.rasterGrid], int)
            brack = Render.__gridPerPrimitive(corner, mm, bracket, gridX, gridY)
            return lax.dynamic_update_slice(grid, brack, (0, 0, i, 0)) 

        def inside_grid_loop(i, grid):
            x = i % 6 * Render.rasterGrid
            y = i // 6 * Render.rasterGrid

            inside_loop = partial(inside_primitive_loop, x, y)
            
            bracket = lax.fori_loop(0, corners.shape[0], inside_loop, lax.dynamic_slice(grid, (x, y, 0, 0), (Render.rasterGrid, Render.rasterGrid, corners.shape[0], 4)), unroll=True)
            return lax.dynamic_update_slice(grid, bracket, (x, y, 0, 0))

        return lax.fori_loop(0, bracketY * bracketX, inside_grid_loop, grid, unroll=True)
    
    @jit
    def generateFragments_B(pos3, faces):
        corners = vmap(Render._getCorners, [0, None])(faces, pos3)
        minmax = vmap(Render._getMinMax, [0])(corners)
        return Render._createBracketing(corners, minmax)
