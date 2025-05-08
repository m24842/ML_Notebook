from .transformers import transformers
from .rnns import rnns

def initialize_model(name, *args, **kwargs):
    modules = [
        transformers,
        rnns,
    ]
    for module in modules:
        if hasattr(module, name):
            model_class = getattr(module, name)
            return model_class(*args, **kwargs)