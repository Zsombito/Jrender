
import jax.experimental
import jax.experimental.host_callback
from .scene import Scene
from .object import Model
from .r_types import Float, Integer, BoolV, Position, Face, PosXNorm, Vec3f, Matrix4, Normal, UV, Array
from jax import vmap, jit
import jax
import jax.numpy as jnp
import jax.lax as lax
from typing import Callable, List, Tuple, Any
from .util_functions import homogenousToCartesian


class Render:
    #Scene variables
    scenes : dict[str, Scene] = {}
    rendered_scenes : list[str] = []
    clipping_batch = 4
    currentScene : Scene = None
    rasterGrid = 50 
    faceBatch = 2
    bracketBatch = 2
    fragBatch = 5
    primitiveParalell = 50000

    #Reset pipeline
    @staticmethod
    def flush_pipeline():
        Render.scenes.clear()
        Render.rendered_scenes.clear()
    
    #Handle scenes
    @staticmethod
    def add_Scene(scene : Scene, name : str, addToRender: bool = True):
        """
        Add a scene to the pipeline
        -- scene : The scene to add
        -- name : The name of the scene (can be accesed later via said name)
        -- addToRender : Add it to the list of rendered scenes
        """

        if name in Render.scenes:
            raise Exception(f"A scene with name '{name}' already exists!")
        
        Render.scenes[name] = scene
        if addToRender:
            Render.rendered_scenes.append(name)

        Render.scene_update = True



    @staticmethod      
    def remove_Scene(name : str):
        """
        Remove scene from the list of scenes based on the name of the scene
        """
        if name not in Render.scenes:
            raise Exception(f"The scene with name '{name}' does not exist")

        if name in Render.rendered_scenes:
            Render.rendered_scenes.pop(name)
        
        Render.rendered_scenes.pop(name)
        Render.scene_update = True
        
    @staticmethod    
    def add_Scene_to_render(name : str):
        """
        Add selected scene to the rendered scenes
        """
        if name not in Render.scenes:
            raise Exception(f"The scene with name '{name}' does not exist")

        if name not in Render.rendered_scenes:
            Render.rendered_scenes.add(name)

        Render.scene_update = True

    
    @staticmethod
    def remove_Scene_from_render(name : str):
        """
        Remove selected scene from the rendered scenes
        """
        if name  in Render.rendered_scenes:
            Render.rendered_scenes.pop(name)

        Render.scene_update = True
    
    
    @staticmethod
    def loadVertexShaders(vertexShader : Callable, vertexExtractor : Callable):
        """
        Can modify the vertex shaders
        """
        Render.vertexExtractor = vertexExtractor
        Render.vertexShader = vertexShader

    @staticmethod
    def loadFragmentShaders(fragmentShader : Callable, fragmentShaderExtractor : Callable):
        """
        Can modify the fragment shaders
        """
        Render.fragmentShader = fragmentShader
        Render.fragmentShaderExtractor = fragmentShaderExtractor

        

    def arrayBatcher(limit, array, contentShape , dim = 0,dummyValue = 0):
        """
        Generates minibatches based on the parameters from the given array
        -limit: Items in a single batch
        -array: The array to batch
        -contentShape: Tuple of the shape of the internal datastructure of array
        -dim (0 by default): Which axis to batch along
        -dummyValue(0 by default): What dummy value to use in order to fill the rest of the last mini batch
        """
        shape = array.shape
        size = shape[dim]
        condition = (size % limit) == 0
        numberOfBatches = size // limit
        if not condition:
            numberOfBatches += 1
        fillAmnt = numberOfBatches * limit - size
        ret_Array = jnp.append(array, (jnp.ones((fillAmnt, *contentShape), array.dtype) * dummyValue))
        return ret_Array.reshape(numberOfBatches, limit, *contentShape), fillAmnt

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
    def _geometryStage():
        """
        Internal function to execute the whole clipping stage, including vertex shaders, clipping and view transformation
        Return value as a tuple:
        - face_mask: This will be the mask for filtering the clipped faces
        - (pos3, norm, face): A tuple including the 3D postions, the normals and the faces
        - perVertexExtra: Extra vertex information extracted by the vertex extractor shader
        - shaded_perVertexExtra: Extra vertex information generated by the vertex shader
        """
        camera = Render.currentScene.camera 

        with jax.named_scope("Vertex shading"):
            (pos, norm, face), perVertexExtra, shaded_perVertexExtra  = Render._applyVertexShader()
        
        with jax.named_scope("Clipping"):
            face_mask = Render._clip(pos, face, camera.fov, camera.aspect)
        
        with jax.named_scope("Viewport transform"):
            pos3 = Render._viewPort(pos, camera.viewPortMatrix)
        
        return face_mask, (pos3, norm, face), perVertexExtra, shaded_perVertexExtra

        

###################################################################
#                        Bracketing                               #
###################################################################

    def _faceBatching(face, pos3):
        """
        Special version of minibatching for primitives, in order to add dummy vertex as well
        """
        pos3 = jnp.append(pos3, jnp.ones((1, 3), float) * -1).reshape(pos3.shape[0] + 1, 3)
        batched_faces, _ = Render.arrayBatcher(Render.faceBatch, face, [3], 0, pos3.shape[0])
        return pos3, batched_faces


    @jit
    def __getCorners(face : Face, position : Float[Vec3f, "idx"]):
        corners = jnp.array([position[face[0]], position[face[1]], position[face[2]]])
        return corners

    @jit
    def __getMinMax(corners):
        minX = corners[:, 0].min()
        maxX = corners[:, 0].max()
        minY = corners[:, 1].min()
        maxY = corners[:, 1].max()
        return jnp.array([minX, maxX, minY, maxY], int)

        
    @jit
    def __createBracketsMasks(brackX, brackY, batched_minmaxs):
        def mapPrimitive(x, y, minmax):
            return (minmax[0] <= (x+1) * Render.rasterGrid) & (x * Render.rasterGrid <= minmax[1]) & (minmax[2] <= (y+1) * Render.rasterGrid) & (y * Render.rasterGrid <= minmax[3])

        def mapBatch(x, y, minmaxs):
            return vmap(mapPrimitive, [None, None, 0])(x, y, minmaxs)
            
        def mapY(x, y, batched_minmaxs):
            return vmap(mapBatch, [None, None, 0])(x, y, batched_minmaxs)

        def mapX(x, brackY, batched_minmaxs):
            return vmap(mapY, [None, 0, None])(x, brackY, batched_minmaxs)


        return vmap(mapX, [0, None, None])(brackX, brackY, batched_minmaxs)

    @jit 
    def __createBrackets(brackX, brackY, batched_corners):

        def mapPrimitive(x, y, batchidx, idx):
            return batchidx * Render.faceBatch + idx

        def mapBatch(x, y, batchidx, corners):
            return vmap(mapPrimitive, [None, None, None, 0])(x, y, batchidx, lax.iota(int, corners.shape[0]))

        def mapY(x, y, batched_corners):
            return vmap(mapBatch, [None, None, 0, 0])(x, y, lax.iota(int, batched_corners.shape[0]), batched_corners)

        def mapX(x, brackY, batched_corners):
            return vmap(mapY, [None, 0, None])(x, brackY, batched_corners)

        return vmap(mapX, [0, None, None])(brackX, brackY, batched_corners)

    @jit
    def _bracketing(pos3, batched_face):

        def mapCornersAlongBatches(faces, pos3):
            return vmap(Render.__getCorners, [0, None])(faces, pos3)       
        
        
        def mapMinmaxAlongBatches(corners):
            return vmap(Render.__getMinMax, [0])(corners)       


        batched_corners = vmap(mapCornersAlongBatches, [0, None])(batched_face, pos3)       #Get the corrseponding 3 vertecies for each primitive
        batched_minmax = vmap(mapMinmaxAlongBatches, [0])(batched_corners)                  #Get minimum and maximum positions for each primitive
        
        #Calculate the necessary parameters for the brackets
        camera = Render.currentScene.camera
        bracketX = camera.X // Render.rasterGrid + 1
        bracketY = camera.Y // Render.rasterGrid + 1
        brackX = lax.iota(int, bracketX)
        brackY = lax.iota(int, bracketY)



        brackets = Render.__createBrackets(brackX, brackY, batched_corners)
        bracket_mask = Render.__createBracketsMasks(brackX, brackY, batched_minmax)
        return batched_corners, brackets, bracket_mask

###################################################################
#                        Rasterization                            #
###################################################################
    @jit
    def __interpolatePrimitive(x, y, idx, corners):
        corners = corners[idx]
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

        beta = (d11 * d20 - d01 * d21) / denom
        gamma = (d00 * d21 - d01 * d20) / denom
        alpha = 1.0 - gamma - beta
        depth = alpha * corners[0, 2] + beta * corners[1, 2] + gamma * corners[2, 2]
        keep = (alpha >= 0) & (beta >= 0) & (gamma >= 0)
        depth = jnp.where(keep, depth, jnp.inf)

        return jnp.array([alpha, beta, gamma, idx, depth], float)


    @jit
    def _rasterize(brackets, x, y, corners, batchnum):
        def mapBatch(x, y, brackets):
            return vmap(Render.__interpolatePrimitive, [None, None, 0, None])(x, y, brackets, corners)

        def mapY(x, y, batched_brackets):
            return vmap(mapBatch, [None, None, 0])(x, y, batched_brackets)

        def mapX(x, gridY, batched_brackets):
            return vmap(mapY, [None, 0, None])(x, gridY, batched_brackets)
        
        gridX = lax.iota(int, Render.rasterGrid) + (jnp.ones(Render.rasterGrid, int) * x * Render.rasterGrid)
        gridY = lax.iota(int, Render.rasterGrid) + (jnp.ones(Render.rasterGrid, int) * y * Render.rasterGrid)


        fragment_candidates = vmap(mapX, [0, None, None])(gridX, gridY, brackets)
        
        

        def mapYZTest(frag_depths, x, y, fragment_candidates):
            idx = frag_depths.argmin()
            return fragment_candidates[x, y, idx // Render.bracketBatch, idx % Render.bracketBatch, :]
        
        def mapXZTest(frag_depths, x, gridY, fragment_candidates):
            return vmap(mapYZTest, [0,None, 0, None])(frag_depths, x, gridY, fragment_candidates)
        


        gridX = lax.iota(int, Render.rasterGrid) 
        gridY = lax.iota(int, Render.rasterGrid)

        selected_fragments = vmap(mapXZTest, [0, 0, None, None])(fragment_candidates[:, :, :, :, 4], gridX, gridY, fragment_candidates)
        

        return selected_fragments

        

#####################################################################
#                     Fragment Shading                              #
#####################################################################

    @jit 
    def _fragmentShading(fragments, faces, norms, perVertexExtra, shaded_perVertexExtra):
        def mapY(fragment):
            idx = fragment[3].astype(int)
            primitiveData, modelID = Render.currentScene.fragmentExtractor(idx, faces, norms, perVertexExtra, shaded_perVertexExtra)
            diffText = Render.currentScene.diffuseText[modelID]
            specText = Render.currentScene.specText[modelID]
            return Render.currentScene.fragmentShader(fragment, Render.currentScene.lights, diffText, specText,  *primitiveData)
        
        
        def mapX(fragments):
            return vmap(mapY, [0])(fragments)

        return vmap(mapX, [0])(fragments)



#####################################################################
#                       Buffer mixing                               #
#####################################################################


    @jit
    def _bufferMixing(shaded_fragments, frameBuffer, x, y):
        def mapY(shaded_fragment, defaultV):
            return jnp.where(shaded_fragment[0] == jnp.inf, defaultV, shaded_fragment[1:])

        def mapX(shaded_fragments, defaultVs):
            return vmap(mapY, [0,0])(shaded_fragments, defaultVs)
        
        sub_area = lax.dynamic_slice(frameBuffer, [x * Render.rasterGrid, y * Render.rasterGrid, 0], [50, 50, 3])
        new_area = vmap(mapX, [0, 0])(shaded_fragments, sub_area)
        return lax.dynamic_update_slice(frameBuffer, new_area, [x * Render.rasterGrid, y * Render.rasterGrid, 0])
            
        


#####################################################################
#                     Main Pipeline Control                         #
#####################################################################

    

    @staticmethod
    def render_C():
        for scene in Render.rendered_scenes:
            Render.currentScene = Render.scenes[scene]
            with jax.named_scope("Geometry Stage:"):
                face_mask, (pos3, norm, face), perVertexExtra, shaded_perVertexExtra = Render._geometryStage()

            print(f"Amount of faces:{face.shape}")

            with jax.named_scope("Face filtering and batchin"):
                face = face[face_mask, :]
                pos3, batchedFaces  = Render._faceBatching(face, pos3)

            with jax.named_scope("Create brackets"):
                corners, brackets, bracket_mask = Render._bracketing(pos3, batchedFaces)
                corners = corners.reshape(corners.shape[0] * corners.shape[1], 3, 3)
                print(f"The shape of the bracket_mask:{bracket_mask.shape}")

            frame_buffer = jnp.zeros((brackets.shape[0] * Render.rasterGrid, brackets.shape[1] * Render.rasterGrid, 3), float)

            for i in range(brackets.shape[0]):
                for j in range(brackets.shape[1]):
                    with jax.named_scope("Filter Brackets"):
                        curr_bracket = brackets[i, j, :, :]
                        curr_mask = bracket_mask[i, j, :, :]
                        if curr_mask.sum() == 0:
                            continue
                        curr_bracket, _ = Render.arrayBatcher(Render.bracketBatch, curr_bracket[curr_mask], [], 0, corners.shape[0] - 1)
                    
                    with jax.named_scope("Rasterization"):
                        fragments = Render._rasterize(curr_bracket, i, j, corners, curr_bracket.shape[0])

                    with jax.named_scope("Fragment shading"): #Continue from here
                        shaded_fragments = Render._fragmentShading(fragments, face, norm, perVertexExtra, shaded_perVertexExtra)
                        (shaded_fragments)

                    with jax.named_scope("Buffer mixing"):
                        frame_buffer = Render._bufferMixing(shaded_fragments, frame_buffer, i, j)

            frame_buffer = frame_buffer * 255
            return frame_buffer[:Render.currentScene.camera.X, :Render.currentScene.camera.Y, :].astype(int)

                        

                
            
