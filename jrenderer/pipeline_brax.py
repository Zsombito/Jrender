import time
import jax.experimental
import jax.experimental.host_callback
from .scene import Scene
from .camera import Camera
from .r_types import Float, Integer, BoolV, Position, Face, Vec3f, Matrix4
from jax import vmap, jit
import jax
import jax.numpy as jnp
import jax.lax as lax
from typing import Callable
from .util_functions import homogenousToCartesian


class Render:
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
    def _applyVertexShader(scene : Scene, camera : Camera):
        #Extract vertex information
        args, argAxis, face, perVertexExtra = Render.vertexExtractor(scene, camera)
        (pos, norm), shaded_PerVertexExtra = vmap(Render.vertexShader, argAxis)(*args)
        return ((pos, norm, face), perVertexExtra, shaded_PerVertexExtra)

        

    @jit
    def __clipVertex(position : Position, near : float, far : float, top : Vec3f, bot : Vec3f, left : Vec3f, right : Vec3f) :
        pos3D = homogenousToCartesian(position)
        return ((pos3D[2] > near) & (pos3D[2] < far) & (jnp.dot(pos3D, top) < 0) & (jnp.dot(pos3D, bot) < 0) & (jnp.dot(pos3D, left) < 0) & (jnp.dot(pos3D, right) < 0))
    
    @jit
    def __getFrustrumParams(near : float, far: float, fov: float, aspect: float):
        hw = jnp.tan(jnp.pi * aspect * fov/360) * near
        hh = hw * (aspect)
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
    def _geometryStage(scene : Scene, camera : Camera):
        with jax.named_scope("Vertex shading"):
            (pos, norm, face), perVertexExtra, shaded_perVertexExtra  = Render._applyVertexShader(scene, camera)
        
        with jax.named_scope("Clipping"):
            face_mask = Render._clip(pos, face, camera.fov, camera.aspect)
        
        with jax.named_scope("Viewport transform"):
            pos3 = Render._viewPort(pos, camera.viewPortMatrix)
        
        return face_mask, (pos3, norm, face), perVertexExtra, shaded_perVertexExtra

        

###################################################################
#                        Bracketing                               #
###################################################################

    def _faceBatching(face, pos3, batch_size):
        """
        Special version of minibatching for primitives, in order to add dummy vertex as well
        """
        pos3 = jnp.append(pos3, jnp.ones((1, 3), float) * -1).reshape(pos3.shape[0] + 1, 3)
        batched_faces, _ = Render.arrayBatcher(batch_size, face, [3], 0, pos3.shape[0])
        return pos3, batched_faces


    @jit
    def __getCorners(face : Face, position : Float[Vec3f, "idx"]):
        corners = jnp.array([position[face[0]], position[face[1]], position[face[2]]])
        return corners


    @jit
    def _generate_corners(pos3, batched_face):

        def mapCornersAlongBatches(faces, pos3):
            return vmap(Render.__getCorners, [0, None])(faces, pos3)       

        batched_corners = vmap(mapCornersAlongBatches, [0, None])(batched_face, pos3)       #Get the corrseponding 3 vertecies for each primitive


        return batched_corners

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
        keep = (alpha > 0) & (beta > 0) & (gamma > 0)
        depth = jnp.where(keep, depth, jnp.inf)

        return jnp.array([alpha, beta, gamma, idx, depth], float)

    def _lineRasterizer_unjitted(gridIdx, gridX, gridY, loop_unroll, corners):
        def mapY(x, y, gridIdx):
            return vmap(Render.__interpolatePrimitive, [None, None, 0, None ])(x, y, gridIdx, corners)

        def mapYZTest(fragment_depths, y, fragment_candidates):
            idx = fragment_depths.argmin()
            return fragment_candidates[y, idx, :]

        def _perRow(_, x):
            fragment_candidates = vmap(mapY, [None, 0, None])(x, gridY, gridIdx)

            selected_fragments = vmap(mapYZTest, [0, 0, None])(fragment_candidates[:, :, 4], gridY, fragment_candidates)

            return None, selected_fragments
        

        _, selected_frags = lax.scan(_perRow, None, gridX, unroll=loop_unroll)
        return selected_frags
    
    _lineRasterizer = jit(_lineRasterizer_unjitted, static_argnames=["loop_unroll"])
        

#####################################################################
#                     Fragment Shading                              #
#####################################################################

    @jit 
    def _fragmentShading(fragments, faces, norms, perVertexExtra, shaded_perVertexExtra, scene : Scene, camera : Camera):
        def mapY(fragment):
            idx = fragment[3].astype(int)
            primitiveData, modelID = Render.fragmentShaderExtractor(idx, faces, norms, perVertexExtra, shaded_perVertexExtra)
            diffText = scene.diffuseText[modelID]
            specText = scene.specText[modelID]
            return Render.fragmentShader(fragment, scene.lights, camera.position, diffText, specText,  *primitiveData)
        
        
        def mapX(fragments):
            return vmap(mapY, [0])(fragments)

        return vmap(mapX, [0])(fragments)



#####################################################################
#                       Buffer mixing                               #
#####################################################################


    @jit
    def _bufferMixing(shaded_fragments, frameBuffer):
        def mapY(shaded_fragment, defaultV):
            return jnp.where(shaded_fragment[0] == jnp.inf, defaultV, shaded_fragment[1:])

        def mapX(shaded_fragments, defaultVs):
            return vmap(mapY, [0,0])(shaded_fragments, defaultVs)
        
        return vmap(mapX, [0, 0])(shaded_fragments, frameBuffer)



#####################################################################
#                     Main Pipeline Control                         #
#####################################################################

    @staticmethod
    def render_forward(scene : Scene, camera : Camera, loop_unroll = 100, debug = False, clipping_batch_size = 300):
        """
        Renders the image based on the scene and camera parameters.
        Optional parameters:
        - loop_unroll: Controls the loop_unroll in the rasterizing stage (default value is 50)
        - debug: Allows for debug prints
        - clipping_batch_size: Allows users to set the exact clipping_batch_size (default value is 300)
        """
        with jax.named_scope("Geometry Stage:"):
            face_mask, (pos3, norm, face), perVertexExtra, shaded_perVertexExtra = Render._geometryStage(scene, camera)

        start = time.time_ns()
        with jax.named_scope("Face filtering and batchin"):
            face = face[face_mask, :]
            pos3, batchedFaces  = Render._faceBatching(face, pos3, clipping_batch_size)
        end = time.time_ns()

        with jax.named_scope("Create brackets"):
            corners = Render._generate_corners(pos3, batchedFaces)
            corners = corners.reshape(corners.shape[0] * corners.shape[1], 3, 3)

        with jax.named_scope("Rasterization"):
            fragments = Render._lineRasterizer(lax.iota(int, corners.shape[0]), camera.pixelsX, camera.pixelsY, loop_unroll, corners)

        with jax.named_scope("Fragment shading"): #Continue from here
            shaded_fragments = Render._fragmentShading(fragments, face, norm, perVertexExtra, shaded_perVertexExtra, scene, camera)

        with jax.named_scope("Buffer mixing"):
            frame_buffer = Render._bufferMixing(shaded_fragments, camera.defaultFrame)

        frame_buffer = frame_buffer * 255
        if debug:
            return frame_buffer.astype(int), (end - start) / 1000 / 1000
        else:
            return frame_buffer.astype(int)



    @jit
    def render_by_parts_GeometryStage(scene : Scene, camera : Camera):
        with jax.named_scope("Geometry Stage:"):
            face_mask, (pos3, norm, face), perVertexExtra, shaded_perVertexExtra = Render._geometryStage(scene, camera)
        
        return (face_mask, face, pos3), norm, perVertexExtra, shaded_perVertexExtra 

    
    def render_by_parts_Filtering(face_mask, face, pos3 ):
        with jax.named_scope("Face filtering and batchin"):
            face = face[face_mask, :]
            pos3, batchedFaces  = Render._faceBatching(face, pos3)
        
        return batchedFaces, pos3, face

    @jit
    def render_by_part_Rasterization(batchedFaces, pos3, face, norm, perVertexExtra, shaded_perVertexExtra, scene, camera):
        with jax.named_scope("Create brackets"):
            corners = Render._generate_corners(pos3, batchedFaces)
            corners = corners.reshape(corners.shape[0] * corners.shape[1], 3, 3)

        with jax.named_scope("Rasterization"):
            fragments = Render._lineRasterizer(lax.iota(int, corners.shape[0]), camera.pixelsX, camera.pixelsY, 1, corners)

        with jax.named_scope("Fragment shading"): #Continue from here
            shaded_fragments = Render._fragmentShading(fragments, face, norm, perVertexExtra, shaded_perVertexExtra, scene, camera)

        with jax.named_scope("Buffer mixing"):
            frame_buffer = Render._bufferMixing(shaded_fragments, camera.defaultFrame)

        frame_buffer = frame_buffer * 255
        return frame_buffer.astype(int)

