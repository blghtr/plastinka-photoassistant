import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from .image_processing import debug
from .account_management import manage_users


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
    registration_page = st.Page(register, title="Register", icon=":material/account_circle:")
    account_management_page = st.Page(manage_users, title="Управление пользователями", icon=":material/manage_accounts:")
    image_processing_page = st.Page(debug, title="Обработка изображений", icon=":material/image:", default=True)

    if st.session_state['authentication_status']:
        pg = st.navigation(
            [image_processing_page, account_management_page, logout_page],
        )

    else:
        pg = st.navigation(
            [login_page, registration_page],
        )

    return pg