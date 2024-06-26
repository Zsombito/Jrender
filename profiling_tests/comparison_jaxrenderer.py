
import functools
from typing import Iterable, NamedTuple, Optional

import jax
from jax import numpy as jp
import numpy as onp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image

import brax
from brax import base, envs, math
from brax.envs import humanoid
from brax.io import model
from brax.training.acme import running_statistics, specs
import brax.training.agents.ppo.networks as ppo_networks
import brax.training.agents.sac.networks as sac_networks

import trimesh

from renderer import CameraParameters as Camera
from renderer import LightParameters as Light
from renderer import Model as RendererMesh
from renderer import ModelObject as Instance
from renderer import ShadowParameters as Shadow
from renderer import Renderer, UpAxis, create_capsule, create_cube, transpose_for_display
from jrenderer.brax_adaptor import BraxRenderer
from jrenderer.pipeline_brax_without_clipping import Render
from jrenderer.shader import stdVertexExtractor, stdVertexShader, stdFragmentExtractor, stdFragmentShader
import time



canvas_width: int = 128 #@param {type:"integer"}
canvas_height: int = 128 #@param {type:"integer"}

def grid(grid_size: int, color) -> jp.ndarray:
  grid = onp.zeros((grid_size, grid_size, 3), dtype=onp.single)
  grid[:, :] = onp.array(color) / 255.0
  grid[0] = onp.zeros((grid_size, 3), dtype=onp.single)
  # to reverse texture along y direction
  grid[:, -1] = onp.zeros((grid_size, 3), dtype=onp.single)
  return jp.asarray(grid)

_GROUND: jp.ndarray = grid(100, [200, 200, 200])

class Obj(NamedTuple):
  """An object to be rendered in the scene.

  Assume the system is unchanged throughout the rendering.

  col is accessed from the batched geoms `sys.geoms`, representing one geom.
  """
  instance: Instance
  """An instance to be rendered in the scene, defined by jaxrenderer."""
  link_idx: int
  """col.link_idx if col.link_idx is not None else -1"""
  off: jp.ndarray
  """col.transform.rot"""
  rot: jp.ndarray
  """col.transform.rot"""

def _build_objects(sys: brax.System) -> list[Obj]:
    """Converts a brax System to a list of Obj."""
    objs: list[Obj] = []

    for geom_idx in range(sys.ngeom):
      tex = sys.geom_rgba[geom_idx][:3].reshape((1, 1, 3))
      # reference: https://github.com/erwincoumans/tinyrenderer/blob/89e8adafb35ecf5134e7b17b71b0f825939dc6d9/model.cpp#L215
      specular_map = jax.lax.full(tex.shape[:2], 2.0)
      geom_type = sys.geom_type[geom_idx]
      if geom_type == 6: #Box
          model = create_cube(sys.geom_size[geom_idx][0], tex, specular_map)
      elif geom_type == 2: #Sphere
          model = create_capsule(sys.geom_size[geom_idx][0], 0, 2, tex, specular_map)
      elif geom_type == 3: #Capsule
          if sys.geom_size[geom_idx].shape[0] == 1:
              model = create_capsule(sys.geom_size[geom_idx][0], 1 * sys.geom_size[geom_idx][0], 2, tex, specular_map)
          else:
              model = create_capsule(sys.geom_size[geom_idx][0], sys.geom_size[geom_idx][1], 2, tex, specular_map)
      else:
          continue

      instance = Instance(model=model)
      obj = Obj(instance=instance, link_idx=sys.geom_bodyid[geom_idx] - 1, off=sys.geom_pos[geom_idx], rot=sys.geom_quat[geom_idx])
      objs.append(obj)

    return objs


def _with_state(objs: Iterable[Obj], x: brax.Transform, idx = 0) -> list[Instance]:
  """x must has at least 1 element. This can be ensured by calling
    `x.concatenate(base.Transform.zero((1,)))`. x is `state.x`.

    This function does not modify any inputs, rather, it produces a new list of
    `Instance`s.
  """
  instances: list[Instance] = []
  for obj in objs:
    i = obj.link_idx
    pos = x.pos[i] + math.rotate(obj.off, x.rot[i])
    rot = math.quat_mul(x.rot[i], obj.rot)
    instance = obj.instance
    instance = instance.replace_with_position(pos)
    instance = instance.replace_with_orientation(rot)
    instances.append(instance)

  return instances

def get_camera(
    width: int = canvas_width,
    height: int = canvas_height,
) -> Camera:
  """Gets camera object."""
  eye, up = jp.array([10, 10, 10]), jp.array([0,0,1])
  hfov = 58.0
  vfov = hfov * height / width
  target = jp.zeros(3, float)
  camera = Camera(
      viewWidth=width,
      viewHeight=height,
      position=eye,
      target=target,
      up=up,
      hfov=hfov,
      vfov=vfov,
  )

  return camera

@jax.default_matmul_precision("float32")
def render_instances(
  instances: list[Instance],
  width: int,
  height: int,
  camera: Camera,
  light: Optional[Light] = None,
  shadow: Optional[Shadow] = None,
  camera_target: Optional[jp.ndarray] = None,
  enable_shadow: bool = False,
) -> jp.ndarray:
  """Renders an RGB array of sequence of instances.

  Rendered result is not transposed with `transpose_for_display`; it is in
  floating numbers in [0, 1], not `uint8` in [0, 255].
  """
  if light is None:
    direction = jp.array([0, 0, 1])
    light = Light(
        direction=direction,
        ambient=0.8,
        diffuse=0.8,
        specular=0.6,
    )
  if shadow is None and enable_shadow:
    assert camera_target is not None, 'camera_target is None'
    shadow = Shadow(centre=camera_target)
  elif not enable_shadow:
    shadow = None

  img = Renderer.get_camera_image(
    objects=instances,
    light=light,
    camera=camera,
    width=width,
    height=height,
    shadow_param=shadow,
    loop_unroll=50
  )
  arr = jax.lax.clamp(0., img, 1.)

  return arr

human = humanoid.Humanoid()


obj = _build_objects(human.sys)
print("Building done")


import pickle

with open('states.pkl', 'rb') as f:
    states : list[brax.State] = pickle.load(f)

frames = []
camera = get_camera(canvas_width, canvas_height)


timesA = []
timesB = []
configs = [
    {"X" : 16, "Y" : 16},
    {"X" : 64, "Y" : 64},
    {"X" : 256, "Y" : 256},
    {"X" : 1280, "Y" : 720},
]

renderer = BraxRenderer.create(human.sys)

Render.loadVertexShaders(stdVertexShader, stdVertexExtractor)
Render.loadFragmentShaders(stdFragmentShader, stdFragmentExtractor)

for t in range(4):
  
  tmp = []
  for i in range(121):
      instances = jax.block_until_ready(_with_state(obj, states[i].pipeline_state.x))
      start = time.time_ns()
      pixels = jax.block_until_ready(render_instances(instances, configs[t]["X"], configs[t]["Y"], camera))
      end = time.time_ns()
      print(f"Frame{i} took: {(end - start) / 1000 / 1000} ms")
      
      if i != 0:
          tmp.append((end - start) / 1000 / 1000)

  timesA.append(tmp)
  tmp = []

  renderer = renderer.config(configs[t])

  for i in range(121):
      start = time.time_ns()
      pixels = jax.block_until_ready(renderer.renderState(states[i].pipeline_state))
      end = time.time_ns()
      print(f"Frame{i} took: {(end - start) / 1000 / 1000} ms")
      if i != 0:
          tmp.append((end - start) / 1000 / 1000)
    
  timesB.append(tmp)



import matplotlib.pyplot as plt

color = ["green", "blue", "red", "purple"]
frames = range(120)
for i in range(4):
    x, y = configs[i]["X"], configs[i]["Y"]
    plt.plot(frames, timesB[i], label = f"Jrenderer Resolution: {x}x{y}", color=color[i], linestyle="dashed")
    plt.plot(frames, timesA[i], label = f"Jaxrenderer Resolution: {x}x{y}", color=color[i])

plt.legend()
plt.ylabel("Time taken to render resolution (ms)")
plt.xlabel("Frame idx")
plt.suptitle("Rendering Different Resolutions Comparison")
plt.savefig("./tests/ComparisonResolution.png")
plt.savefig("./tests/ComparisonResolution.svg")

pixels = []
avaragesA = []
avaragesB = []
final = []
for i in range(4):
    x, y = configs[i]["X"], configs[i]["Y"]
    pixels.append(x*y)
    avaragesA.append(sum(timesA[i]) / 120)
    avaragesB.append(sum(timesB[i]) / 120)

print(avaragesA)
print(avaragesB)
final.append(avaragesA)
final.append(avaragesB)

plt.clf()
plt.plot(pixels, avaragesA, label = "Jaxrenderer")
plt.plot(pixels, avaragesB, label = "Jrenderer (ours)", linestyle="dashed")
plt.legend()
plt.ylabel("Time taken to render (ms)")
plt.xlabel("Pixel Count")
plt.suptitle("Rendering Different Resolutions Comparison")
plt.savefig("./tests/ComparisonResolutionComp.png")
plt.savefig("./tests/ComparisonResolutionComp.svg")


def render_env(renderer : BraxRenderer, state : brax.State, loop_unroll):
    return renderer.renderState(state, loop_unroll)
    

def _render_batch_unjitted(renderers, state, loop_unroll = 50):
    return jax.vmap(render_env, [0, None, None])(renderers, state, loop_unroll)

render_batch = jax.jit(_render_batch_unjitted, static_argnames=["loop_unroll"])

batch_size = [1, 5, 10, 50]
@jax.jit
def create_renderer(sys : brax.System, Ids : int):
  def _create(sys : brax.System, idx):
    renderer = BraxRenderer.create(sys)
    return renderer.config({"X":128,"Y":128})
  return jax.vmap(_create, [None, 0])(sys, Ids)

@jax.jit
def batch_state(state, Ids):
  def _batch(state, idx):
    return state

  return jax.vmap(_batch, [None, 0])(state, Ids)

@jax.jit
def vmap_statechange(obj, state, ids):
  return jax.vmap(_with_state, [None, None, 0])(obj, state, ids)

@jax.jit
def vmap_jaxrenderer(batched_instances, camera):
  return jax.vmap(render_instances, [0, None, None, None])(batched_instances, canvas_width, canvas_height, camera)

camera = get_camera()
timesA = []
timesB = []
for t in range(4):
  Ids = jax.lax.iota(int, batch_size[t])
  obj = _build_objects(human.sys)
  tmp = []
  for i in range(121):
    batched_states = jax.block_until_ready(vmap_statechange(obj, states[i].pipeline_state.x, Ids))
    start = time.time_ns()
    _ = jax.block_until_ready(vmap_jaxrenderer(batched_states, camera))
    end = time.time_ns()
    if i != 0:
      tmp.append((end - start) / 1000 / 1000)
    print(f"Running test {t}, frame {i}, with jaxrenderer")
  
  timesA.append(tmp)
  renderers = create_renderer(human.sys, Ids)
  tmp = []
  for i in range(121):
    start = time.time_ns()
    _ = jax.block_until_ready(render_batch(renderers, states[i].pipeline_state, 50))
    end = time.time_ns()
    if i != 0:
      tmp.append((end - start) / 1000 / 1000)
    print(f"Running test {t}, frame {i}, with jrenderer")
  timesB.append(tmp)
  
  
  
color = ["green", "blue", "red", "purple"]

plt.clf()
frames = range(120)
for i in range(4):
    plt.plot(frames, timesB[i], label = f"Jrenderer with batch size: {batch_size[i]}" , linestyle="dashed", color=color[i])
    plt.plot(frames, timesA[i], label = f"Jaxrenderer with batch size: {batch_size[i]}", color = color[i])

plt.legend()
plt.ylabel("Time taken to render batch (ms)")
plt.xlabel("Frame idx")
plt.suptitle("Batch Rendering Different Batch Sizes")
plt.savefig("./tests/CompBatchTest.png")
plt.savefig("./tests/CompBatchTest.svg")

final = []
avragesA = []
avragesB = []
for i in range(4):
    avragesB.append(sum(timesB[i]) / 120)
    avragesA.append(sum(timesA[i]) / 120)

final.append(avragesA)
final.append(avragesB)
plt.clf()
plt.plot(batch_size, avragesA, label="Jaxrenderer")
plt.plot(batch_size, avragesB, label="Jrenderer (ours)", linestyle="dashed")
plt.legend()
plt.ylabel("Time taken to render batch (ms)")
plt.xlabel("Batch size")
plt.suptitle("Batch Rendering Different Batch Sizes Comparison")
plt.savefig("./tests/CompBatchComp.png")
plt.savefig("./tests/CompBatchComp.svg")
    
  
print(final)
  



