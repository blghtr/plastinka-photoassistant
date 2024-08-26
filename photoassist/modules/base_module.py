import inspect
from typing import Dict
from numpy import ndarray


class BaseModule:
    def __init__(self, conf_threshold: float = 0.7, save_intermediate_outputs=False, **kwargs):
        sig = inspect.signature(self.__init__)
        args_dict = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
        args_dict.update(kwargs)
        setattr(self, 'args', args_dict)

    def __call__(self, input_data: Dict):
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
        predicted_class_and_conf = input_data.get('class', None)
        if predicted_class_and_conf is not None:
            _, confidence = predicted_class_and_conf
            if confidence > self.args['conf_threshold']:
                return True
        return False

    def _process(self, input_data: Dict) -> Dict:
        raise NotImplementedError

    def _apply_transform(self, *args, **kwargs) -> ndarray:
        raise NotImplementedError
