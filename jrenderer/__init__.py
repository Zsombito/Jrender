from .model import Model
from .capsule import create_capsule
from .cube import create_cube
from .plane import create_plane
from .camera import Camera
from .scene import Scene
from .shader import stdFragmentExtractor, stdFragmentShader, stdVertexExtractor, stdVertexShader
from .brax_adaptor import BraxRenderer as BraxRenderer
from .pipeline_brax_without_clipping import Render
from .brax_adaptor_with_cliping import BraxRenderer as BraxRenderer_with_Clip
from .pipeline_brax import Render as Render_with_Clip
from .lights import Light

Render.loadVertexShaders(stdVertexShader, stdVertexExtractor)
Render.loadFragmentShaders(stdFragmentShader, stdFragmentExtractor)

Render_with_Clip.loadVertexShaders(stdVertexShader, stdVertexExtractor)
Render_with_Clip.loadFragmentShaders(stdFragmentShader, stdFragmentExtractor)


