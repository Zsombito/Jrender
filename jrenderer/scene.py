from .object import Model
from .camera import Camera
from .r_types import Matrix4, Position, Float, Normal, Integer, Face



class Scene:
    unique : int = 0
    def __init__(self, camera : Camera) -> None:
        self.models : dict[int, Model] = {}
        self.camera : Camera  = camera
        pass

    def add_Model(self, mdl : Model) -> int:
        self.models[Scene.unique] = mdl
        Scene.unique += 1
        return Scene.unique - 1


    def transform_Model(self, idx : int, transform : Matrix4) -> None:
        self.models[idx].applyTransform(transform)
    
    def delete_Model(self, idx : int) -> None:
        self.models.pop(idx)
    
        
        
