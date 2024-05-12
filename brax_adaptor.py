
from typing import Iterable, NamedTuple, Optional

import jax
from jax import numpy as jp
import numpy as onp

import brax
from brax import base, envs, math, positional

import trimesh

from jrenderer.camera import Camera
from jrenderer.lights import Light
from jrenderer.capsule import create_capsule
from jrenderer.cube import create_cube
from jrenderer.pipeline import Render
from jrenderer.model import Model
from jrenderer.scene import Scene


def _build_scene(sys: brax.System) -> tuple[Scene, dict[int, int]]:
    pass
    