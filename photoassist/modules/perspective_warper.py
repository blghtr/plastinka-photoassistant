from copy import copy
from typing import Dict
import cv2
import numpy as np
from .base_module import BaseModule


class PerspectiveWarper(BaseModule):
    """Warp image to a rectified perspective given a border (quadrilateral)."""
    def __init__(self, conf_threshold=0.7, interpolation='INTER_CUBIC', save_intermediate_outputs=True, **kwargs):
        interpolation = getattr(cv2, interpolation)
        super().__init__(
            conf_threshold=conf_threshold,
            interpolation=interpolation,
            save_intermediate_outputs=save_intermediate_outputs,
            **kwargs
        )

    # LLM:METADATA
    # :hierarchy: [PhotoAssist | Modules | PerspectiveWarper]
    # :relates-to: calls: "calculate_w_h", calls: "warp_perspective"
    # :rationale: "Rectify the perspective of the region of interest to a flat view."
    # :contract: pre: "border in input_data", post: "image wrapped to bird's-eye view"
    # LLM:END
    def _process(self, input_data: Dict) -> Dict:
        """If 'border' is present, compute target size and warp perspective."""
        if input_data.get('border', None) is None:
            self.logger.warning("'border' not found in input data, skipping perspective warping.")
            return input_data

        self.logger.debug("Border found, proceeding with perspective warp.")
        max_width, max_height = calculate_w_h(input_data['border'])
        self.logger.debug(f"Calculated max_width={max_width}, max_height={max_height}")

        if input_data.get('class', None) is not None:
            class_name = input_data['class'][0]
            if class_name != 'booklet':
                self.logger.debug(f"Class is '{class_name}', not 'booklet'. Making dimensions square.")
                max_width = max_height = max(max_width, max_height)
                self.logger.debug(f"New dimensions: max_width={max_width}, max_height={max_height}")

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
        """Return a copy of the warped image."""
        return copy(input_data['image'])


# LLM:METADATA
# :hierarchy: [PhotoAssist | Modules | PerspectiveWarper | Utils]
# :rationale: "Compute the dimensions of the rectified image based on corner distances."
# :contract: pre: "points is 4x2 array", post: "returns (max_width, max_height)"
# LLM:END
def calculate_w_h(points):
    """Compute target width and height from a quadrilateral's side lengths."""
    a, b, c, d = points
    width_ab = np.sum(np.abs(a - b))
    width_dc = np.sum(np.abs(d - c))
    max_width = int(max(width_ab, width_dc))

    height_ad = np.sum(np.abs(a - d))
    height_cb = np.sum(np.abs(c - b))
    max_height = int(max(height_ad, height_cb))

    return max_width, max_height


# LLM:METADATA
# :hierarchy: [PhotoAssist | Modules | PerspectiveWarper | Utils]
# :relates-to: uses: "cv2.getPerspectiveTransform", uses: "cv2.warpPerspective"
# :rationale: "Transform image region defined by points into a rectangular format."
# :contract: pre: "image and points valid", post: "returns warped image"
# LLM:END
def warp_perspective(image, points, max_width, max_height, interpolation):
    """Apply cv2.warpPerspective to map `points` onto a target rectangle."""
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
