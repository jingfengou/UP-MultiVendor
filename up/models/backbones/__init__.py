# 公共 backbone 注册保留 ResNet/ViT；YOLOX 的专属 backbone 由对应插件注册。
from .resnet import resnet50  # noqa


from .vision_transformer import vit_base_patch16_224  # noqa





from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY

imported_vars = list(globals().items())

for var_name, var in imported_vars:
    if callable(var):
        MODULE_ZOO_REGISTRY.register(var_name, var)
