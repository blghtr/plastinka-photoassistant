# Plastinka Photoassistant

Инструмент для пакетной обработки фотографий конвертов и буклетов виниловых пластинок. Включает веб‑интерфейс на Streamlit с аутентификацией и модульный конвейер обработки изображений: сегментация (YOLOv8), пост‑обработка маски и построение рамки, коррекция экспозиции по ArUco, выравнивание перспективы и ресайз.

## Возможности
- **Веб‑интерфейс**: загрузка изображений, отслеживание прогресса, скачивание архива результатов
- **Аутентификация**: управление пользователями через `streamlit-authenticator`
- **Модульный пайплайн**: лёгкая конфигурация порядка и параметров модулей через YAML
- **Сегментация YOLOv8**: поиск целевого объекта и маски с поддержкой CPU/GPU
- **Пост‑обработка маски**: аппроксимация контура, восстановление скошенных углов, построение границ
- **Коррекция баланса/экспозиции**: по белому полю около метки ArUco
- **Выравнивание перспективы**: трансформация под нужный формат (квадрат/прямоугольник)
- **Ресайз**: масштабирование под заданную длинную сторону
- **Логи и отчёты**: сохранение ошибок и отчётов о выполнении

## Структура проекта
```
photoassist/                 # Пакет с пайплайном и модулями
  modules/
    base_module.py           # Базовый класс модуля пайплайна
    segmenter.py             # Сегментация YOLOv8
    mask_post_processing.py  # Пост‑обработка маски и построение рамки
    balancer.py              # Коррекция баланса по ArUco
    perspective_warper.py    # Выравнивание перспективы
    framer.py                # Формирование кадра по маске/классу
    resizer.py               # Ресайз итогового изображения
    writer.py                # Сохранение результатов, отчёты
  pipeline/
    pipeline.py              # Оркестрация пайплайна, распараллеливание
    config.py                # Обёртка конфигурации YAML

user_interface/
  app.py                     # Конфигурация страниц Streamlit
  image_processing.py        # Основной UI обработки и скачивания
  account_management.py      # Управление пользователями
  my_logging.py              # Настройка логирования

main.py                      # Точка входа для Streamlit (navigation.run())
requirements.txt             # Зависимости Python
default_config.yaml          # Базовая конфигурация пайплайна
debug_config.yaml            # Конфигурация с промежуточными результатами
```

## Требования
- Python 3.10+
- PyTorch (CPU или CUDA по желанию)
- OpenCV с модулями ArUco (`opencv-contrib-python`)
- Ultralytics (YOLOv8)

Установите зависимости:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> Примечание: для GPU установите подходящую сборку `torch` согласно инструкции PyTorch.

## Подготовка весов модели
По умолчанию используется путь к весам из `default_config.yaml`:
```yaml
modules:
  Segmenter:
    model: yolo_v8_s_800_1.pt
```
Скопируйте файл весов `yolo_v8_s_800_1.pt` в корень проекта или укажите абсолютный/относительный путь к файлу в конфигурации.

## Конфигурация
Конфигурация пайплайна хранится в YAML (`default_config.yaml`, `debug_config.yaml`). Структура:
```yaml
modules:
  Segmenter:           # порядок и параметры инициализации модуля
    order: 0
    conf_threshold: 0.5
    model: yolo_v8_s_800_1.pt
    device: cpu            # либо cuda:0
    save_intermediate_outputs: False
  Balancer:
    order: 1
    aruco_dict: DICT_4X4_50
    aruco_idx: 0
    offset: 10
    C: 1
    save_intermediate_outputs: False
  PostProcessor:
    order: 3
    max_dist: 150
    min_angle: 90
    ...
  PerspectiveWarper:
    order: 4
    interpolation: INTER_CUBIC
  Resizer:
    order: 5
    longest_side: 1500

pipeline:
  n_jobs: 1                 # число параллельных задач в joblib
  save_intermediate_outputs: False
```
- `order`: порядок выполнения модулей
- `save_intermediate_outputs`: при True модуль добавит в результат визуализацию своего шага (показывается в UI и полезно для отладки)
- Прочие параметры — специфичны для модуля (см. код в `photoassist/modules/*`).

## Секреты и аутентификация
Используется `streamlit-authenticator`. Перед запуском создайте `st_secrets.yaml` в корне:
```yaml
credentials:
  usernames:
    admin:
      email: admin@example.com
      name: Administrator
      password: "hashed_password"
pre-authorized:
  emails: []
cookie:
  name: some_cookie
  key: some_key
  expiry_days: 30
```
- Создать/обновить пользователей можно на странице «Управление пользователями». Файл `st_secrets.yaml` будет перезаписан.

## Запуск
```bash
streamlit run main.py
```
- Откроется мультистраничный интерфейс:
  - «Обработка изображений»: загрузка файлов, прогресс, скачивание архива
  - «Управление пользователями»: регистрация/редактирование/удаление
  - «Ошибки»: доступно админу, просмотр логов из `logs/errors`

## Использование пайплайна из кода
```python
from PIL import Image
from photoassist import PipelineConfig, Pipeline

config = PipelineConfig('default_config.yaml')
pipeline = Pipeline(config)

inputs = [
  {"image": Image.open("in1.jpg"), "name": "in1.jpg"},
  {"image": Image.open("in2.jpg"), "name": "in2.jpg"},
]
results = pipeline(inputs)
# Каждый элемент: словарь с ключами 'image' (np.ndarray BGR) и 'name'
```

## Логи и отчёты
- Ошибки шагов сохраняются в `logs/errors` (см. `user_interface/my_logging.py` и `Writer`)
- В режиме пакетной обработки `Writer.report()` формирует отчёт со статистикой

## Отладка
Используйте `debug_config.yaml`, где включены промежуточные результаты для всех модулей. Они отображаются в UI для текущего изображения.

## Лицензия
Проект предназначен для внутреннего использования. Уточните условия распространения в вашей организации.