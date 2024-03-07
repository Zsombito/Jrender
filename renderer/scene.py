from .object import Model
from .r_types import Matrix4



class Scene:
    unique : int = 0
    def __init__(self) -> None:
        self.models : dict[int, Model] = {}
        pass

    def add_Model(self, mdl : Model) -> int:
        self.models[Scene.unique] = mdl
        Scene.unique += 1
        return Scene.unique - 1

    def transform_Model(self, idx : int, transform : Matrix4) -> None:
        self.models[idx].applyTransform(transform)
    
    def delete_Model(self, idx : int) -> None:
        self.models.pop(idx)
    

        
        
