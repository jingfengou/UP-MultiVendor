import torch
import torch.nn as nn

from up.utils.general.registry_factory import MODEL_WRAPPER_REGISTRY

__all__ = ['DetWrapper', 'ClsWrapper']


@MODEL_WRAPPER_REGISTRY.register('det')
class DetWrapper(torch.nn.Module):
    def __init__(self, detector):
        super(DetWrapper, self).__init__()
        self.detector = detector

    def forward(self, image, return_metas=False):
        b, c, height, width = map(int, image.size())
        input = {
            'image_info': [[height, width, 1.0, height, width, False, 0, 0]] * b,
            'image': image
        }
        output = self.detector(input)
        blob_names = []
        blob_datas = []
        output_names = sorted(output.keys())
        for name in output_names:
            if name.find('blobs') >= 0:
                blob_names.append(name)
                blob_datas.append(output[name])
        assert len(blob_datas) > 0, 'no valid output provided, please set "tocaffe: True" in your config'
        if return_metas:
            return blob_names
        else:
            return blob_datas


@MODEL_WRAPPER_REGISTRY.register('cls')
class ClsWrapper(torch.nn.Module):
    def __init__(self, detector, add_softmax=False):
        super(ClsWrapper, self).__init__()
        self.detector = detector
        self.add_softmax = add_softmax
        if self.add_softmax:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, image, return_metas=False):
        b, c, height, width = map(int, image.size())
        input = {
            'image_info': [[height, width, 1.0, height, width, False, 0, 0]] * b,
            'image': image
        }
        output = self.detector(input)
        if self.add_softmax:
            if isinstance(output['scores'], list):
                output['scores'] = [self.softmax(score) for score in output['scores']]
            else:
                output['scores'] = self.softmax(output['scores'])
        if isinstance(output['scores'], list):
            # blob_names = ['scores_%d' % i for i in range(len(output['scores']))]
            blob_names = ['scores']
            output['scores'] = torch.cat(output['scores'], dim=1)
        else:
            blob_names = ['scores']
        if return_metas:
            return blob_names
        else:
            return [output['scores']]
