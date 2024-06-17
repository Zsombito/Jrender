import brax
from jrenderer.pipeline import Render
import jax

import brax
from brax.envs import humanoid
from jrenderer.pipeline_brax_without_clipping import Render
from jrenderer.brax_adaptor import BraxRenderer
from jrenderer.shader import stdVertexExtractor, stdVertexShader, stdFragmentExtractor, stdFragmentShader




#Creating Brax Environment
human = humanoid.Humanoid()


#Initializing and configuring the renderer
brax_renderer = BraxRenderer.create(human.sys)
config = {"CamLinkMode":0, "CamLinkTarget" : 1}
brax_renderer = brax_renderer.config(config)

#Loading states
import pickle
with open('states.pkl', 'rb') as f:
    states : list[brax.State] = pickle.load(f)

#Generate frames
frames = []
for i in range(100):
    pixels= jax.block_until_ready(brax_renderer.renderState(states[i].pipeline_state))
    frames.append(pixels)

print("Making giff")
import imageio
imageio.mimsave('./brax_output/output.gif', frames)