from .base_module import BaseModule
import cv2
from typing import Dict


class Framer(BaseModule):
    def __init__(self, conf_threshold: float = 0.5, ksize: int = 15, save_intermediate_outputs: bool = True):
        super().__init__(
            conf_threshold=conf_threshold,
            ksize=ksize,
            save_intermediate_outputs=save_intermediate_outputs
        )

    def _fork(self, input_data: Dict) -> Dict:
        if input_data.get('class', None) is not None:
            class_name = input_data['class'][0]
            if class_name == 'apple':
                return input_data
        return self._process_other(input_data)

    def _process_other(self, input_data: Dict) -> Dict:
        image, mask = input_data['image'], input_data['mask']
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        blurred_mask = cv2.GaussianBlur(mask, (self.args['ksize'], self.args['ksize']), 0)
        norm_blurred_mask = cv2.normalize(blurred_mask, None, 0, 1, cv2.NORM_MINMAX)
        input_data['image'] = cv2.multiply(norm_blurred_mask, image)
        return input_data

    def _process(self, input_data: Dict) -> Dict:
        return self._fork(input_data)

    def _apply_transform(self, input_data):
        return input_data['image']
