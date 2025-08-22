import streamlit as st
import streamlit_authenticator as stauth
import yaml
import time
from .my_logging import get_logger


secrets_path = 'st_secrets.yaml'
logger = get_logger(__name__)


def save_button_click(button_name):
    """Set a flag in session state indicating a button was pressed."""
    st.session_state.button_state[button_name] = True


def reset_button_click(button_name):
    """Reset a previously set button flag in session state."""
    st.session_state.button_state[button_name] = False


def authorize_new_user():
    """Add an email to the pre-authorized allowlist and persist config."""
    config = st.session_state.config

    st.write('Add an e-mail to the allowlist:')
    with st.form("Question", clear_on_submit=True):
        email = st.text_input("Enter e-mail:")
        submitted = st.form_submit_button("Submit")
        if submitted and email is not None:
            st.success('E-mail added to allowlist')
            config['pre-authorized']['emails'].append(email)
            with open(secrets_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            time.sleep(1)
            st.rerun()


def manage_users():
    """User management page for editing and deleting users."""
    st.session_state.button_state = {}

    authenticator = st.session_state._authenticator
    user = st.session_state['username']
    config = st.session_state.config

    authenticator.login(location='unrendered')

    users = ['New user']
    users.extend(config['credentials']['usernames'].keys())

    choice = user if user != 'admin' else st.sidebar.selectbox("Choose user", users)
    st.title('User Management')
    if choice is not None:
        if choice == 'New user':
            authorize_new_user()
        else:
            st.markdown(
                f"""
               ### User data:
                - **Username**: {choice}
                - **Name**: {config['credentials']['usernames'][choice]['name']}
                - **Email**: {config['credentials']['usernames'][choice]['email']}
                """
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.button_state.get("edit_user", False) or \
                        st.button(
                            "Edit",
                            key="edit_user",
                            on_click=save_button_click,
                            kwargs={"button_name": "edit_user"}
                        ):
                    try:
                        if authenticator.update_user_details(choice):
                            st.success('Updated successfully')
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
                        st.error('An error occurred, try again later or contact the admin')

            with col2:
                if st.session_state.button_state.get("delete_user", False) or \
                        st.button(
                            "Delete",
                            key="delete_user",
                            help="Only admin can delete a user",
                            on_click=save_button_click,
                            kwargs={"button_name": "delete_user"},
                            disabled=choice == user
                        ):
                    credentials = authenticator.authentication_controller.authentication_model.credentials
                    o = credentials['usernames'].pop(choice, None)
                    if o is not None:
                        st.success('Deleted successfully')
                        config['credentials'] = credentials
                        with open(secrets_path, 'w') as file:
                            yaml.dump(config, file, default_flow_style=False)
                        st.session_state.config = config

                    else:
                        st.error('Deletion error')
                    reset_button_click("delete_user")
                    st.rerun()

