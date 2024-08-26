from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Union, Dict, List, Optional
import cv2
from .base_module import BaseModule

CLASS_NAMES = ['apple', 'vinyl_envelops', 'booklet']
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

class Writer(BaseModule):
    def __init__(
            self,
            log_dir: Union[str, PathLike],
            out_dir: Union[str, PathLike],
            overwrite: bool = True,
            optional_modules: Optional[List[str]] = None
    ):

        out_dir = Path(out_dir) / f'{TIMESTAMP}'
        log_dir = Path(log_dir) / f'{TIMESTAMP}'
        not_processed_dir = out_dir / 'not_processed'
        errors_dir = log_dir / 'errors'
        class_dirs = [out_dir / class_name for class_name in CLASS_NAMES]

        for dir in [out_dir, log_dir, not_processed_dir, errors_dir, *class_dirs]:
            dir.mkdir(parents=True, exist_ok=True)

        if optional_modules is not None:
            optional_modules = set(optional_modules)

        super().__init__(
            log_dir=log_dir,
            out_dir=out_dir,
            not_processed_dir=not_processed_dir,
            errors_dir=errors_dir,
            class_dirs=class_dirs,
            overwrite=overwrite,
            optional_modules=optional_modules,
            timestamp=TIMESTAMP
        )


    def __call__(self, input_data: Dict) -> Dict:
        if 'exc_tb' in input_data:
            input_data['result'] = False
            return input_data

        steps_applied = [v for k, v in input_data['steps_applied'].items() if k not in self.args['optional_modules']]
        processed = all(steps_applied)

        class_name = input_data['class'][0]
        if processed:
            path = Path(self.args['out_dir'] / class_name / str(Path(input_data['orig_path']).name))
        else:
            path = Path(self.args['not_processed_dir'] / str(Path(input_data['orig_path']).name))

        result = False

        if not path.exists() or self.args['overwrite']:
            result = cv2.imwrite(str(path), input_data['image'])
        return {'result': result, 'path': str(path)}

    def report(self, result, single_input=False, errors_count=0):
        if not single_input:
            not_processed_dir = self.args['not_processed_dir']
            not_processed_count = len(list(Path(not_processed_dir).glob('*')))

            formatted_report = f"""
            ===============================
                     ОТЧЕТ
            ===============================

            Всего изображений в источнике: {len(result)}
            ------------------------------------
            Не обработано в связи с 
            недостаточно точной сегментацией: {not_processed_count}

            Найти эти изображения можно в папке: {not_processed_dir}
            ------------------------------------
            Всего ошибок в процессе обработки: {errors_count}

            Смотрите логи в папке: {self.args['errors_dir']}
            """
            with open(
                    Path(
                        self.args['log_dir']) / f'report_{self.args["timestamp"]}.txt',
                    'w',
                    encoding='utf-8'
            ) as f:
                f.write(formatted_report)
        self._save_errors(result)

    def _save_errors(self, result):
        errors = filter(lambda x: 'exc_tb' in x, result)
        for err in errors:
            self._write_single_error(err)

    def _write_single_error(self, error):
        write_path = Path(self.args['errors_dir']) / f'error_item_{error["item_path"].stem}_{self.args["timestamp"]}.txt'
        with open(write_path, 'w', encoding='utf-8') as f:
            f.write(f"Error processing item: {error['item_path']}\n")
            for line in error['exc_tb']:
                f.write(line)
                f.write('\n')





