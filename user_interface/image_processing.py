import streamlit as st
from PIL import Image
from photoassist import PipelineConfig, Pipeline


config = PipelineConfig('debug_config.yaml')
pipeline = Pipeline(config)


def process_images():
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
            print(result)
            for m_name, result_image in result['intermediate_outputs'].items():
                st.image(result_image, caption=f'Результат {m_name}', use_column_width=True)


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
                                      accept_multiple_files=True)

    if uploaded_files:
        st.session_state.uploaded_images = [Image.open(file).convert('RGB') for file in uploaded_files]
        if st.button('Выполнить предсказание'):
            process_images()

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