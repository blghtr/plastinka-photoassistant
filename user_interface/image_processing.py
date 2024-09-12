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
from copy import copy


logger = get_logger(__name__)
config = PipelineConfig('default_config.yaml')
pipeline = Pipeline(config)


def process_images_debug():
    st.session_state.results = []
    inp = copy(st.session_state.uploaded_images)
    results = pipeline(inp[::-1])
    st.session_state.results.extend(results)


def show_current_image():
    if st.session_state.uploaded_images:
        image = st.session_state.uploaded_images[st.session_state.current_image_index]['image']
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
        input_data = list(
            map(lambda x: {'image': Image.open(x), 'name': x.name}, uploaded_files)
        )

        st.session_state.uploaded_images = input_data
        if st.button('Выполнить предсказание'):
            process_images_debug()

    if st.session_state.results:
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


def create_zip(data, progress_bar):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as zip_file:
        data_len = c = len(data)
        while c:
            elem = data.pop()
            img, img_name = elem['image'], elem['name']
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="JPEG", quality=100)
            zip_file.writestr(img_name, img_buffer.getvalue())
            c -= 1
            progress_bar.progress((data_len - c) / data_len, "Результаты архивируются...")
    return zip_buffer.getvalue()


def process_images():
    def reset_uploader():
        st.session_state.results = []
        st.session_state.uploaded_images = []
        st.session_state.uploader_key += 1
        gc.collect()

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
        my_bar = st.progress(0.)

        uploaded_files = list(
            map(lambda x: {'image': Image.open(x), 'name': x.name}, uploaded_files)
        )

        st.session_state.uploaded_images = uploaded_files

        if st.session_state.uploaded_images:
            pipeline.set_callback('progress_tracker', my_bar.progress)
            all_results = pipeline(st.session_state.uploaded_images)
            results = list(filter(lambda x: 'exc_tb' not in x, all_results))
            results = list(
                map(
                    lambda x: {
                        'image': Image.fromarray(cv2.cvtColor(
                            x['image'],
                            cv2.COLOR_BGR2RGB
                        ).astype(np.uint8)),
                        'name': x['name']
                    },
                    results
                )
            )
            st.session_state.results = results
            st.session_state.errors = list(filter(lambda x: 'exc_tb' in x, all_results))

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
            archive = create_zip(st.session_state.results, my_bar)
            if st.download_button(
                label="Скачать архив",
                data=archive,
                file_name="images.zip",
                mime="application/zip",
                on_click=reset_uploader
            ):
                del archive
    elif uploaded_files is not None:
        st.error('Загрузите изображения, прежде чем приступить к обработке')
