from .common import *
from .rnns import *
from .transformers import *

from . import rnns
from . import transformers
from . import *

def initialize_model(name, *args, **kwargs):
    modules = [
        transformers,
        rnns,
    ]
    for module in modules:
        if hasattr(module, name):
            model_class = getattr(module, name)
            return model_class(*args, **kwargs)