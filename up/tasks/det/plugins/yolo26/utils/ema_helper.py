"""Official YOLO26 EMA exposed through UP's existing EMA registry."""
from copy import deepcopy

from up.utils.general.registry_factory import EMA_REGISTRY
from ..models.model import official_model


@EMA_REGISTRY.register('official_yolo26')
class YOLO26EMA:
    def __init__(self, model):
        from ultralytics.utils.torch_utils import ModelEMA
        # Keep the same UP model/weight keys around the official EMA network.
        self.model = deepcopy(model.module if hasattr(model, 'module') else model).eval()
        self.official = ModelEMA(official_model(model).network)
        official_model(self.model).network = self.official.ema

    def step(self, model, curr_step=None):
        self.official.update(official_model(model).network)

    def state_dict(self):
        return {'ema_state_dict': self.model.state_dict(), 'updates': self.official.updates}

    def load_state_dict(self, state):
        self.model.load_state_dict(state['ema_state_dict'])
        self.official.updates = state['updates']
