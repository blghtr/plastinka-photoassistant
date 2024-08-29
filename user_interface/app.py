import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from .image_processing import process_images
from .account_management import manage_users
from pathlib import Path


error_path = Path('logs/errors')
if not error_path.exists():
    error_path.mkdir(parents=True)
secrets_path = 'st_secrets.yaml'


def login():
    st.title('Plastinka Photoassistant')
    st.session_state._authenticator.login()
    if st.session_state['authentication_status']:
        pass


def logout():
    st.session_state._authenticator.logout(location='unrendered')
    st.rerun()


def register():
    config = st.session_state.config
    authenticator = st.session_state._authenticator
    try:
        email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(
            pre_authorization=True)
        if email_of_registered_user:
            st.success('Регистрация успешна')
            config['credentials'] = authenticator.authentication_controller.authentication_model.credentials
            config['pre-authorized'] = authenticator.authentication_controller.authentication_model.pre_authorized
            with open(secrets_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
    except Exception as e:
        st.error(e)

def errors():
    st.session_state._authenticator.login(location='unrendered')
    all_errors = ['Ошибка...']
    all_errors.extend([error.name for error in error_path.glob('*.txt')])

    error_filename = st.sidebar.selectbox('Выберите ошибку', all_errors)
    if error_filename is not None and error_filename != 'Ошибка...':
        with open(error_path / error_filename) as file:
            st.json(yaml.safe_load(file.read()))

def get_app():
    with open(secrets_path) as file:
        config = yaml.load(file, Loader=SafeLoader)

    st.session_state.config = config
    st.session_state._authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['pre-authorized']
    )

    login_page = st.Page(login, title="Log in", icon=":material/login:")
    logout_page = st.Page(logout, title="Log out", icon=":material/logout:")
    errors_page = st.Page(errors, title="Errors", icon=":material/error:")
    registration_page = st.Page(register, title="Register", icon=":material/account_circle:")
    account_management_page = st.Page(manage_users, title="Управление пользователями", icon=":material/manage_accounts:")
    image_processing_page = st.Page(process_images, title="Обработка изображений", icon=":material/image:", default=True)

    if st.session_state['authentication_status']:
        pages = [image_processing_page, account_management_page]
        if st.session_state['username'] == 'admin':
            pages.append(errors_page)
        pages.append(logout_page)
        pg = st.navigation(pages)

    else:
        pg = st.navigation(
            [login_page, registration_page],
        )

    return pg