from collections import OrderedDict
from os import PathLike
from typing import Union, Dict
import ultralytics
from numpy import ndarray
import numpy as np
from torch import as_tensor
from .base_module import BaseModule


class Segmenter(BaseModule):
    def __init__(
            self,
            model: Union[str, PathLike],
            conf_threshold: float = 0.5,
            device: str = 'cpu',
            save_intermediate_outputs: bool = True
    ):
        self.model = ultralytics.YOLO(model)

        super().__init__(
            model=model,
            conf_threshold=conf_threshold,
            device=device,
            save_intermediate_outputs=save_intermediate_outputs
        )

    def __call__(self, input_data: Dict):
        return self.segment(input_data)

    def segment(self, input_data: Dict) -> Dict:
        image = input_data['image']
        results = self.model.predict(source=image, max_det=1, device=self.args['device'], retina_masks=True)
        save_intermediate_output = input_data.get('intermediate_outputs', None) \
                                   and self.args['save_intermediate_outputs']

        result = cut_mask(results[0])
        return {
            'image': result.orig_img,
            'mask': result.masks.cpu().xy[0],
            'box': result.boxes.cpu().xyxy.numpy()[0],
            'class': (result.names[(int(result.boxes.cls.item()))], result.boxes.conf.item()),
            'orig_path': result.path,
            'intermediate_outputs': OrderedDict(
                [(self.__class__.__name__, self._apply_transform(result))]
            ) if save_intermediate_output else None
        }

    def _apply_transform(self, result) -> ndarray:
        if result.names[(int(result.boxes.cls.item()))] == 'apple':
            return result.plot(masks=False)
        return result.plot(boxes=True, line_width=5)


def cut_mask(result):
    mask_obj = result.masks.cpu()
    mask = mask_obj.data.numpy().transpose((1, 2, 0))
    new_mask = np.zeros_like(mask)
    h, w, _ = new_mask.shape
    bbx = result.boxes.xyxyn.numpy()[0]
    x_min, y_min, x_max, y_max = (bbx * np.array([w, h, w, h])).astype(np.int32)
    new_mask[y_min:y_max, x_min:x_max] = mask[y_min:y_max, x_min:x_max]
    result.update(masks=as_tensor(new_mask.transpose((2, 0, 1))))
    return result