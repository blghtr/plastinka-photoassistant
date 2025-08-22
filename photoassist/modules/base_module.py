import inspect
from typing import Dict
from numpy import ndarray


class BaseModule:
    """Base class for all pipeline modules.

    Contract:
    - __call__(input_data) checks confidence threshold and calls _process
    - _process(input_data) must be implemented by subclasses and return updated dict or None
    - _apply_transform(...) returns a visualization for intermediate outputs
    - _conf_check(...) checks confidence from input_data['class'] -> (label, confidence)
    """
    def __init__(self, **kwargs):
        setattr(self, 'args', kwargs)

    def __call__(self, input_data: Dict):
        """Runs the module respecting confidence threshold and saving intermediates."""
        input_data.setdefault('steps_applied', {})[self.__class__.__name__] = False
        enough_conf = self._conf_check(input_data)
        if not self.args['conf_threshold'] or enough_conf:
            out = self._process(input_data)
            if out is None:
                return input_data

            if input_data.get('intermediate_outputs', None) is not None and self.args['save_intermediate_outputs']:
                input_data['intermediate_outputs'][self.__class__.__name__] = self._apply_transform(
                    out
                )

            input_data['steps_applied'][self.__class__.__name__] = True
            return out
        return input_data

    def _conf_check(self, input_data: Dict) -> bool:
        """Return True if confidence in input_data['class'] exceeds the threshold."""
        predicted_class_and_conf = input_data.get('class', None)
        if predicted_class_and_conf is not None:
            _, confidence = predicted_class_and_conf
            if confidence > self.args['conf_threshold']:
                return True
        return False

    def _process(self, input_data: Dict) -> Dict:
        """Core module logic. Must be implemented by subclasses."""
        raise NotImplementedError

    def _apply_transform(self, *args, **kwargs) -> ndarray:
        """Return an image/visualization for intermediate outputs."""
        raise NotImplementedError
