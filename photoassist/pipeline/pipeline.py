import importlib
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List
from joblib import Parallel, delayed, cpu_count
import traceback
from PIL.Image import Image


class Pipeline:

    def __init__(self, config: 'PipelineConfig'):
        self.modules_config = self._load_modules(config['modules'])
        self.config = config['pipeline']


    def _load_modules(self, config):
        modules = OrderedDict()
        py_module = importlib.import_module('photoassist.modules')
        ordered_config = sorted(config.items(), key=lambda x: x[1]['order'])
        for module_name, init_params in ordered_config:
            _ = init_params.pop('order', None)
            module = getattr(py_module, module_name)
            modules[module_name] = {
                'module': module,
                'init_params': init_params
            }
        return modules

    def _init_modules(self, config=None):
        if config is None:
            config = self.modules_config
        return [module_dict['module'](**module_dict['init_params']) for module_dict in config.values()]

    def __call__(self, images: List[Image]) -> List[Dict]:
        return self._run(images)

    def _run(self, images: List[Image]):
        def _run_in_parallel():
            modules = self._init_modules()
            return Parallel(n_jobs=n_jobs)(
                delayed(_process_modules)(result, modules) for result in minibatch
            )

        n_jobs = self.config['n_jobs'] if self.config['n_jobs'] > 0 else cpu_count()
        all_results = []
        minibatch = []
        n_images = len(images)
        while n_images:
            input_data = {'image': images.pop()}
            minibatch.append(input_data)
            n_images -= 1
            if len(minibatch) == n_jobs:
                all_results.extend(_run_in_parallel())
                minibatch.clear()
        if len(minibatch):
            all_results.extend(_run_in_parallel())
        return all_results


class ProcessingErrorHandler(Exception):
    def __init__(self, input_data, step):
        super().__init__()
        self.error_occurred = False
        self.item_path = Path(input_data['orig_path']) if input_data is not None and 'orig_path' in input_data else None
        self.step = step

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_value is not None:
            tb = traceback.format_exception(exc_type, exc_value, exc_tb)
            self.error_occurred = True
            self.tb = tb
        return True

    def get_result(self):
        return [
            {
                'path': self.item_path,
                'module': self.step,
                'exc_tb': self.tb
            }
        ]


def _process_modules(input_data, modules):
    if len(modules):
        for module in modules:
            with ProcessingErrorHandler(input_data, module.__class__.__name__) as error_handler:
                input_data = module(input_data)
            if error_handler.error_occurred:
                return error_handler.get_result()
    return input_data
