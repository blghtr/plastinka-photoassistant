import inspect
from typing import Dict
from numpy import ndarray


class BaseModule:
	"""Базовый класс модуля пайплайна.

	Контракт:
	- __call__(input_data) выполняет проверку порога уверенности и вызывает _process
	- _process(input_data) реализуется в наследнике и возвращает обновлённый словарь либо None
	- _apply_transform(...) возвращает визуализацию шага для промежуточных результатов
	- _conf_check(...) проверяет порог уверенности по полю 'class' -> (label, confidence)
	"""
	def __init__(self, **kwargs):
		setattr(self, 'args', kwargs)

	def __call__(self, input_data: Dict):
		"""Запускает модуль с учётом порога уверенности и сохранения промежуточных результатов."""
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
		"""Возвращает True, если confidence из input_data['class'] превышает порог."""
		predicted_class_and_conf = input_data.get('class', None)
		if predicted_class_and_conf is not None:
			_, confidence = predicted_class_and_conf
			if confidence > self.args['conf_threshold']:
				return True
		return False

	def _process(self, input_data: Dict) -> Dict:
		"""Основная логика модуля. Должна быть реализована в наследниках."""
		raise NotImplementedError

	def _apply_transform(self, *args, **kwargs) -> ndarray:
		"""Возвращает изображение/визуализацию для промежуточных результатов."""
		raise NotImplementedError
