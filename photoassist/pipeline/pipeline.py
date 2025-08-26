import importlib
import logging
from collections import OrderedDict, defaultdict
from pathlib import PurePath
from typing import Dict, List, Optional, Callable
from joblib import Parallel, delayed, cpu_count
import traceback
import gc


class Pipeline:
    """Modular image-processing pipeline.

    Executes a configured sequence of modules over a list of inputs.
    Supports mini-batch parallelization and progress callbacks.

    Args:
        config: Pipeline configuration (see PipelineConfig)
        callbacks: Optional mapping of callbacks, e.g. {'progress_tracker': callable}
        logger: Optional logger instance.
    """

    def __init__(self, config: 'PipelineConfig', callbacks: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        else:
            self.logger = logger

        self.modules_config = self._load_modules(config['modules'])
        self.config = config['pipeline']
        self.progress = 0
        self.callbacks = defaultdict(None)
        if callbacks is not None:
            for name, callback in callbacks.items():
                self.set_callback(name, callback)
        self.logger.info("Pipeline initialized.")

    def _load_modules(self, config):
        """Load and order modules by 'order' field.

        Returns an OrderedDict describing modules and their init params.
        """
        self.logger.debug("Loading modules...")
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
            self.logger.debug(f"Loaded module '{module_name}' with params: {init_params}")
        return modules

    def _init_modules(self, config=None):
        """Instantiate modules with parameters from the configuration."""
        if config is None:
            config = self.modules_config
        
        initialized_modules = []
        for module_dict in config.values():
            # Pass logger to each module
            module_params = module_dict['init_params'].copy()
            module_params['logger'] = self.logger.getChild(module_dict['module'].__name__)
            initialized_modules.append(module_dict['module'](**module_params))
        return initialized_modules

    def __call__(self, input_data: List[Dict]) -> List[Dict]:
        """Run the pipeline over the provided input list."""
        self.logger.info(f"Starting pipeline run for {len(input_data)} items.")
        results = self._run(input_data)
        self.logger.info("Pipeline run finished.")
        return results

    def _run(self, input_data: List[Dict]) -> List[Dict]:
        """Process a list of items, updating progress and invoking a callback.

        Items look like {'image': np.ndarray|PIL.Image, 'name': str, ...}.
        Returns a list of results or error descriptions.
        """
        def _process_minibatch():
            self.logger.info(f"Processing minibatch of size {len(minibatch)}...")
            all_results.extend(
                Parallel(n_jobs=n_jobs)(
                    delayed(_process_modules)(
                        result, self._init_modules(), self.logger
                    ) for result in minibatch
                )
            )
            minibatch.clear()
            self.progress = len(all_results) / n_images
            self.logger.info(f"Pipeline progress: {self.progress:.2%}")
            if 'progress_tracker' in self.callbacks:
                self.callbacks['progress_tracker'](self.progress, 'Обработка фотографий...')
            gc.collect()

        all_results = []
        minibatch = []
        n_images = c = len(input_data)
        batch_size = max(1, n_images // 10)
        n_jobs = min(batch_size, self.config['n_jobs'])
        
        # Process images in original order (pop(0) extracts from beginning)
        while c:
            minibatch.append(input_data.pop(0))  # Use pop(0) to maintain original order
            c -= 1
            if len(minibatch) == batch_size:
                _process_minibatch()

        if len(minibatch):
            _process_minibatch()

        return all_results

    def set_callback(self, name: str, callback: Callable):
        """Register a callback, e.g., a progress tracker."""
        self.callbacks[name] = callback


class ProcessingErrorHandler(Exception):
    """Context manager to capture a processing step error.

    Stores the filename and traceback, and returns a unified error dict.
    """
    def __init__(self, input_data, step, logger):
        super().__init__()
        self.error_occurred = False
        self.filename = PurePath(input_data['name']).name if input_data is not None else "Unknown"
        self.step = step
        self.logger = logger

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_value is not None:
            tb = traceback.format_exception(exc_type, exc_value, exc_tb)
            self.error_occurred = True
            self.tb = tb
            self.logger.error(f"Error processing '{self.filename}' in module '{self.step}': {''.join(tb)}")
        return True

    def get_result(self):
        """Return an error description for logging and UI display."""
        return {
                'name': self.filename,
                'module': self.step,
                'exc_tb': self.tb
        }


def _process_modules(input_data, modules, logger):
    """Run a single item through the module sequence.

    On error, return the dict produced by ProcessingErrorHandler instead of a result.
    """
    item_name = input_data.get('name', 'Unknown')
    logger.debug(f"Processing item: {item_name}")
    if len(modules):
        for module in modules:
            with ProcessingErrorHandler(input_data, module.__class__.__name__, logger) as error_handler:
                input_data = module(input_data)
                gc.collect()
            if error_handler.error_occurred:
                logger.warning(f"Stopping processing for item {item_name} due to error in {module.__class__.__name__}.")
                return error_handler.get_result()
    logger.debug(f"Finished processing item: {item_name}")
    return input_data
