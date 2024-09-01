import streamlit as st
import streamlit_authenticator as stauth
import yaml
import time
from .my_logging import get_logger


secrets_path = 'st_secrets.yaml'
logger = get_logger(__name__)


def save_button_click(button_name):
    st.session_state.button_state[button_name] = True


def reset_button_click(button_name):
    st.session_state.button_state[button_name] = False


def authorize_new_user():
    config = st.session_state.config

    st.write('Добавьте e-mail в белый список:')
    with st.form("Question", clear_on_submit=True):
        email = st.text_input("Введитe e-mail:")
        submitted = st.form_submit_button("Отправить")
        if submitted and email is not None:
            st.success('E-mail добавлен в белый список')
            config['pre-authorized']['emails'].append(email)
            with open(secrets_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            time.sleep(1)
            st.rerun()

def manage_users():
    st.session_state.button_state = {}

    authenticator = st.session_state._authenticator
    user = st.session_state['username']
    config = st.session_state.config

    authenticator.login(location='unrendered')

    users = ['Новый пользователь']
    users.extend(config['credentials']['usernames'].keys())

    choice = user if user != 'admin' else st.sidebar.selectbox("Выберите пользователя", users)
    st.title('User Management')
    if choice is not None:
        if choice == 'Новый пользователь':
            authorize_new_user()
        else:
            st.markdown(
                f"""
               ### Данные пользователя:
                - **Юзернейм**: {choice}
                - **Имя**: {config['credentials']['usernames'][choice]['name']}
                - **Почта**: {config['credentials']['usernames'][choice]['email']}
                """
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.button_state.get("edit_user", False) or \
                        st.button(
                            "Редактировать",
                            key="edit_user",
                            on_click=save_button_click,
                            kwargs={"button_name": "edit_user"}
                        ):
                    try:
                        if authenticator.update_user_details(choice):
                            st.success('Успешно обновлено')
                            config['credentials'] = authenticator.authentication_controller.authentication_model.credentials
                            with open(secrets_path, 'w') as file:
                                yaml.dump(config, file, default_flow_style=False)
                            st.session_state.config = config
                            reset_button_click("edit_user")
                            st.rerun()

                    except stauth.UpdateError as e:
                        st.error(e)

                    except Exception as e:
                        logger.error(e, exc_info=True)
                        st.error('Произошла ошибка, повторите попытку позже или обратитесь к администратору')

            with col2:
                if st.session_state.button_state.get("delete_user", False) or \
                        st.button(
                            "Удалить",
                            key="delete_user",
                            help="Удалить пользователя может только администратор",
                            on_click=save_button_click,
                            kwargs={"button_name": "delete_user"},
                            disabled=choice == user
                        ):
                    credentials = authenticator.authentication_controller.authentication_model.credentials
                    o = credentials['usernames'].pop(choice, None)
                    if o is not None:
                        st.success('Успешно удален')
                        config['credentials'] = credentials
                        with open(secrets_path, 'w') as file:
                            yaml.dump(config, file, default_flow_style=False)
                        st.session_state.config = config

                    else:
                        st.error('Ошибка удаления')
                    reset_button_click("delete_user")
                    st.rerun()

