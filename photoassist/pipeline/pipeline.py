import importlib
from collections import namedtuple, OrderedDict
from os import PathLike
from pathlib import Path
from typing import Union, Generator
from joblib import Parallel, delayed, cpu_count
import traceback


class Pipeline:

    def __init__(self, config: 'PipelineConfig'):
        self.modules_config = self._load_modules(config['modules'])
        self.config = config['pipeline']
        segmenter_config = self.modules_config.pop('Segmenter', None)
        self._segmenter = self._init_modules({'Segmenter': segmenter_config})[0]
        self._single_input = None
        self._save_on_disk = 'Writer' in self.modules_config

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

    def __call__(self, path: Union[str, PathLike]):
        input_data = {'source': path}
        self._single_input = isinstance(path, str) and not Path(path).is_dir() or not isinstance(path, str)
        if self.config['save_intermediate_outputs'] and self._single_input:
            input_data['intermediate_outputs'] = True
        segmenter = self._segmenter
        with ProcessingErrorHandler(input_data, segmenter.__class__.__name__) as error_handler:
            generator = segmenter(input_data)
        if error_handler.error_occurred:
            return error_handler.get_result()

        run_method = self._run_image if self._single_input else self._run_folder
        results = run_method(generator)

        return results

    def _run_folder(self, generator: Generator):
        def _run_in_parallel():
            modules = self._init_modules()
            return Parallel(n_jobs=self.config['n_jobs'])(
                delayed(_process_modules)(result, modules) for result in samples
            )

        n_jobs = self.config['n_jobs'] if self.config['n_jobs'] > 0 else cpu_count()
        all_results = []
        samples = []
        for sample in generator:
            samples.append(sample)
            if len(samples) == n_jobs:
                all_results.extend(_run_in_parallel())
                samples.clear()
        if len(samples):
            all_results.extend(_run_in_parallel())

        return all_results

    def _run_image(self, generator: Generator):
        image = next(generator)
        modules = self._init_modules()
        result = _process_modules(image, modules)
        return result

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
