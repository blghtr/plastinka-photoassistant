from copy import copy
from typing import Dict, Tuple
import cv2
from numpy import ndarray
from .base_module import BaseModule


class Resizer(BaseModule):
	"""Resize image proportionally so that the longest side equals the target."""
	def __init__(self, longest_side, conf_threshold=0.7, interpolation='INTER_CUBIC', save_intermediate_outputs=True):
		interpolation = getattr(cv2, interpolation)
		super().__init__(
			longest_side=longest_side,
			conf_threshold=conf_threshold,
			interpolation=interpolation,
			save_intermediate_outputs=save_intermediate_outputs
		)

	def _process(self, input_data: Dict) -> Dict:
		"""Resize input_data['image'] keeping aspect ratio."""
		input_data['image'] = cv2.resize(
			input_data['image'],
			self._calculate_proportional_size(input_data),
			interpolation=self.args['interpolation']
		)
		return input_data

	def _calculate_proportional_size(self, input_data: Dict) -> Tuple[int, int]:
		"""Compute new (width, height) preserving aspect ratio by longest side."""
		h, w = input_data['image'].shape[:2]
		max_dim = max(h, w)
		resize_scale = self.args['longest_side'] / max_dim
		return int(w * resize_scale), int(h * resize_scale)

	def _apply_transform(self, input_data: Dict) -> ndarray:
		"""Return a copy of the resized image."""
		return copy(input_data['image'])
