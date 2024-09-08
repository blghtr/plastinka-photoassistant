from copy import copy
from typing import Dict
import cv2
import numpy as np
from .base_module import BaseModule


class PerspectiveWarper(BaseModule):
    def __init__(self, conf_threshold=0.7, interpolation='INTER_CUBIC', save_intermediate_outputs=True):
        interpolation = getattr(cv2, interpolation)
        super().__init__(
            conf_threshold=conf_threshold,
            interpolation=interpolation,
            save_intermediate_outputs=save_intermediate_outputs
        )

    def _process(self, input_data: Dict) -> Dict:
        if input_data.get('border', None) is None:
            return input_data

        max_width, max_height = calculate_w_h(input_data['border'])
        if input_data.get('class', None) is not None:
            class_name = input_data['class'][0]
            if class_name != 'booklet':
                max_width = max_height = max(max_width, max_height)

        input_data['image'] = warp_perspective(
            input_data['image'],
            input_data['border'],
            max_width,
            max_height,
            interpolation=self.args['interpolation']
        )

        del input_data['border']
        return input_data

    def _apply_transform(self, input_data: Dict) -> np.ndarray:
        return copy(input_data['image'])


def calculate_w_h(points):
    a, b, c, d = points
    width_ab = np.sum(np.abs(a - b))
    width_dc = np.sum(np.abs(d - c))
    max_width = int(max(width_ab, width_dc))

    height_ad = np.sum(np.abs(a - d))
    height_cb = np.sum(np.abs(c - b))
    max_height = int(max(height_ad, height_cb))

    return max_width, max_height


def warp_perspective(image, points, max_width, max_height, interpolation):
    output_pts = np.float32(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ]
    )
    M = cv2.getPerspectiveTransform(points, output_pts)
    out = cv2.warpPerspective(image, M, (max_width, max_height), flags=interpolation)
    del M

    return out
