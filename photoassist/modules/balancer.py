from copy import copy
from typing import Dict, Optional
import cv2
import numpy as np
from .base_module import BaseModule


class Balancer(BaseModule):
    def __init__(
            self,
            conf_threshold: float = 0.0,
            aruco_dict: str = 'DICT_4X4_50',
            aruco_idx: int = 0,
            offset: int = 10,
            C: int = -1,
            save_intermediate_outputs: bool = True
    ):
        aruco_dict = getattr(cv2.aruco, aruco_dict)
        super().__init__(
            conf_threshold=conf_threshold,
            aruco_dict=aruco_dict,
            aruco_idx=aruco_idx,
            offset=offset,
            C=C,
            save_intermediate_outputs=save_intermediate_outputs
        )

    def _process(self, input_data: Dict) -> Dict:
        image = np.uint8(copy(input_data['image']))
        corners = get_aruco_corners(image, self.args['aruco_dict'], self.args['aruco_idx'])
        if corners is None:
            return None
        white_x, white_y = corners[:, 0].max(), corners[:, 1].max()
        x1, x2, y1, y2 = (
            white_x + self.args['offset'],
            white_x + 2 * self.args['offset'],
            white_y + self.args['offset'],
            white_y + 2 * self.args['offset']
        )
        white = image[y1:y2, x1:x2]
        target_white = np.mean(white, axis=(0, 1))
        coefficients = target_white.mean() / target_white
        if self.args['C'] >= 0:
            coefficients = 1 + self.args['C'] * (coefficients - 1)
            balanced_image = image * coefficients
            balanced_image = np.clip(balanced_image, 0, 255).astype(np.uint8)
        input_data['image'] = balanced_image
        return input_data

    def _apply_transform(self, input_data: Dict) -> np.ndarray:
        return copy(input_data['image'])


def get_aruco_corners(
        image: np.ndarray,
        aruco_dict: int,
        aruco_idx: int,
    ) -> Optional[np.ndarray]:

    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    marker_corners, marker_ids, _ = detector.detectMarkers(image)
    if marker_ids is not None:
        marker_corners = [c for idx, c in zip(marker_ids, marker_corners) if idx == aruco_idx]
    if len(marker_corners) == 0:
        return None
    int_corners = np.intp(marker_corners).squeeze()
    return int_corners


