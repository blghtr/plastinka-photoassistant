import importlib
from collections import OrderedDict, defaultdict
from pathlib import PurePath
from typing import Dict, List, Optional, Callable
from joblib import Parallel, delayed, cpu_count
import traceback
import gc


class Pipeline:

    def __init__(self, config: 'PipelineConfig', callbacks: Optional[Dict] = None):
        self.modules_config = self._load_modules(config['modules'])
        self.config = config['pipeline']
        self.progress = 0
        self.callbacks = defaultdict(None)
        if callbacks is not None:
            for name, callback in callbacks.items():
                self.set_callback(name, callback)

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

    def __call__(self, input_data: List[Dict]) -> List[Dict]:
        return self._run(input_data)

    def _run(self, input_data: List[Dict]) -> List[Dict]:
        def _process_minibatch():
            all_results.extend(
                Parallel(n_jobs=n_jobs)(
                    delayed(_process_modules)(
                        result, self._init_modules()
                    ) for result in minibatch
                )
            )
            minibatch.clear()
            self.progress = len(all_results) / n_images
            if 'progress_tracker' in self.callbacks:
                self.callbacks['progress_tracker'](self.progress, 'Обработка фотографий...')
            gc.collect()

        all_results = []
        minibatch = []
        n_images = c = len(input_data)
        batch_size = max(1, n_images // 10)
        n_jobs = min(batch_size, self.config['n_jobs'])
        while c:
            minibatch.append(input_data.pop())
            c -= 1
            if len(minibatch) == batch_size:
                _process_minibatch()

        if len(minibatch):
            _process_minibatch()

        return all_results

    def set_callback(self, name: str, callback: Callable):
        self.callbacks[name] = callback


class ProcessingErrorHandler(Exception):
    def __init__(self, input_data, step):
        super().__init__()
        self.error_occurred = False
        self.filename = PurePath(input_data['name']).name if input_data is not None else None
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
        return {
                'name': self.filename,
                'module': self.step,
                'exc_tb': self.tb
        }


def _process_modules(input_data, modules):
    if len(modules):
        for module in modules:
            with ProcessingErrorHandler(input_data, module.__class__.__name__) as error_handler:
                input_data = module(input_data)
                gc.collect()
            if error_handler.error_occurred:
                return error_handler.get_result()
    return input_data
