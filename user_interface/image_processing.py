import streamlit as st
from PIL import Image
from photoassist import PipelineConfig, Pipeline
import numpy as np
import io
import zipfile
import cv2
import gc
from datetime import datetime
from .my_logging import get_logger


logger = get_logger(__name__)
config = PipelineConfig('default_config.yaml')
pipeline = Pipeline(config)


def process_images_debug():
    st.session_state.results = []
    for image in st.session_state.uploaded_images:
        result = pipeline(image)
        st.session_state.results.append(result)


def show_current_image():
    if st.session_state.uploaded_images:
        image = st.session_state.uploaded_images[st.session_state.current_image_index]
        st.image(image, caption=f'Изображение {st.session_state.current_image_index + 1}', use_column_width=True)

        if st.session_state.results:
            result = st.session_state.results[st.session_state.current_image_index]
            for m_name, result_image in result['intermediate_outputs'].items():
                i = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                st.image(i, caption=f'Результат {m_name}', use_column_width=True)


def next_image():
    if st.session_state.current_image_index < len(st.session_state.uploaded_images) - 1:
        st.session_state.current_image_index += 1


def prev_image():
    if st.session_state.current_image_index > 0:
        st.session_state.current_image_index -= 1

def debug():
    # Инициализация состояния сессии
    if 'current_image_index' not in st.session_state:
        st.session_state.current_image_index = 0
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []
    if 'results' not in st.session_state:
        st.session_state.results = []
    st.session_state._authenticator.login(location='unrendered')

    uploaded_files = st.file_uploader("Выберите изображения...", type=["jpg", "jpeg", "png"],
                                      accept_multiple_files=True, key='efser')

    if uploaded_files:
        st.session_state.uploaded_images = [Image.open(file) for file in uploaded_files]
        if st.button('Выполнить предсказание'):
            process_images_debug()

    if st.session_state.uploaded_images:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('Предыдущее', on_click=prev_image):
                pass
        with col2:
            st.write(
                f'Изображение {st.session_state.current_image_index + 1} из {len(st.session_state.uploaded_images)}')
        with col3:
            if st.button('Следующее', on_click=next_image):
                pass
        show_current_image()


def create_zip(data):
    zip_buffer = io.BytesIO()
    unpacked = [(elem['image'], elem['name']) for elem in data]
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for img, img_name in unpacked:
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            zip_file.writestr(img_name, img_buffer.getvalue())
    return zip_buffer.getvalue()


def process_images():
    def reset_uploader():
        st.session_state.results = []
        st.session_state.uploaded_images = []
        st.session_state.uploader_key += 1

    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    if 'errors' not in st.session_state:
        st.session_state.errors = []
    if 'datetime' not in st.session_state:
        st.session_state.datetime = datetime.now().strftime('%Y%m%d_%H%M%S')

    st.session_state._authenticator.login(location='unrendered')

    with st.form('uploader_form', clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Выберите изображения...",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f'uploader_{st.session_state.uploader_key}'
        )
        submitted = st.form_submit_button('Запустить обработку')

    if submitted and uploaded_files is not None and uploaded_files:
        st.session_state.uploaded_images = uploaded_files

        progress_text = "Обработка продолжается..."
        my_bar = st.progress(0., text=progress_text)

        input_data = list(
            map(lambda x: {'image': Image.open(x), 'name': x.name}, st.session_state.uploaded_images)
        )

        st.session_state.uploaded_images = input_data

        if input_data:
            pipeline.set_callback('progress_tracker', my_bar.progress)
            results = pipeline(input_data)
            filtered_results = list(filter(lambda x: 'exc_tb' not in x, results))
            results = list(
                map(
                    lambda x: {
                        'image': Image.fromarray(cv2.cvtColor(
                            x['image'],
                            cv2.COLOR_BGR2RGB
                        ).astype(np.uint8)),
                        'name': x['name']
                    },
                    filtered_results
                )
            )
            st.session_state.results = results
            st.session_state.errors = list(filter(lambda x: 'exc_tb' in x, results))

            if st.session_state.errors:
                for err in st.session_state.errors:
                    name, module, exc_tb = err['name'], err['module'], err['exc_tb']
                    err_message = '\n'.join(
                        [f'Error processing image {name} with {module}:',
                         *exc_tb]
                    )

                    logger.error(err_message)
                st.session_state.errors = []

        if st.session_state.results:
            archive = create_zip(st.session_state.results)
            st.download_button(
                label="Скачать архив",
                data=archive,
                file_name="images.zip",
                mime="application/zip",
                on_click=reset_uploader
            )
    elif uploaded_files is not None:
        st.error('Загрузите изображения, прежде чем приступить к обработке')

    gc.collect()