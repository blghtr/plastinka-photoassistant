import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from .image_processing import process_images, debug_mode
from .account_management import manage_users
from .my_logging import get_logger, ERROR_PATH


logger = get_logger(__name__)
secrets_path = 'st_secrets.yaml'


# LLM:METADATA
# :hierarchy: [UserInterface | Auth]
# :relates-to: uses: "streamlit_authenticator.Authenticate.login"
# :rationale: "Control access to the application via user authentication."
# :contract: pre: "authenticator initialized", post: "updates session_state status"
# LLM:END
def login():
    """Render login page and show auth status messages."""
    st.title('Plastinka Photoassistant')
    st.session_state._authenticator.login()
    if st.session_state['authentication_status'] is False:
        st.error('Incorrect username or password')

    elif st.session_state['authentication_status'] is None:
        st.warning('Please enter your credentials')


# LLM:METADATA
# :hierarchy: [UserInterface | Auth]
# :relates-to: uses: "streamlit_authenticator.Authenticate.logout"
# :rationale: "Terminate user session securely."
# :contract: pre: "user logged in", post: "session cleared and app reruns"
# LLM:END
def logout():
    """Trigger logout and rerun app to return to auth pages."""
    st.session_state._authenticator.logout(location='unrendered')
    st.rerun()


# LLM:METADATA
# :hierarchy: [UserInterface | Auth]
# :relates-to: uses: "streamlit_authenticator.Authenticate.register_user"
# :rationale: "Onboard new users with email verification against allowlist."
# :contract: pre: "authenticator ready", post: "new user credentials saved"
# LLM:END
def register():
    """Render registration flow using streamlit-authenticator."""
    config = st.session_state.config
    authenticator = st.session_state._authenticator
    try:
        (email_of_registered_user,
         username_of_registered_user,
         name_of_registered_user) = authenticator.register_user(
            pre_authorized=config['pre-authorized']['emails'],
            fields={
                'Password': ('Password: 8-20 chars with letters, digits, '
                             '@$!%*?&. Must include upper, lower, digit, '
                             'special.')
            }
        )
        if email_of_registered_user:
            st.success('Registration successful')
            config['credentials'] = (
                authenticator.authentication_controller
                .authentication_model.credentials
            )
            config['pre-authorized'] = (
                authenticator.authentication_controller
                .authentication_model.pre_authorized
            )
            with open(secrets_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)

    except stauth.RegisterError as e:
        st.error(e)

    except Exception as e:
        logger.error(e, exc_info=True)
        st.error('Registration failed. Try again or contact admin')


# LLM:METADATA
# :hierarchy: [UserInterface | Admin]
# :rationale: "Provide visibility into application errors for maintenance."
# :contract: pre: "user is admin", post: "displays log content"
# LLM:END
def errors():
    """Admin page to view error logs written by the app."""
    st.session_state._authenticator.login(location='unrendered')
    all_errors = ['Error...']
    all_errors.extend([
        error.name for error in ERROR_PATH.glob('*') if error.is_file()
    ])
    error_filename = st.sidebar.selectbox('Choose error', all_errors)
    if error_filename is not None and error_filename != 'Error...':
        with open(ERROR_PATH / error_filename) as file:
            errors = file.readlines()
            st.write(errors)


# LLM:METADATA
# :hierarchy: [UserInterface | EntryPoint]
# :relates-to: uses: "streamlit.navigation"
# :rationale: "Configure application routing based on authentication state."
# :contract: pre: "config_path valid", post: "returns streamlit Page object"
# LLM:END
def get_app(config_path: str):
    """Create Streamlit navigation and return the current page object."""
    st.session_state.config_path = config_path
    with open(secrets_path) as file:
        config = yaml.load(file, Loader=SafeLoader)

    st.session_state.config = config
    st.session_state._authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        auto_hash=False
    )

    login_page = st.Page(login, title="Login", icon=":material/login:")
    logout_page = st.Page(logout, title="Logout", icon=":material/logout:")
    errors_page = st.Page(errors, title="Errors", icon=":material/error:")
    registration_page = st.Page(
        register, title="Register", icon=":material/account_circle:"
    )
    account_management_page = st.Page(
        manage_users, title="User Management",
        icon=":material/manage_accounts:"
    )
    image_processing_page = st.Page(
        process_images, title="Image Processing",
        icon=":material/image:", default=True
    )
    debug_page = st.Page(
        debug_mode, title="Debug Mode", icon=":material/bug_report:"
    )

    if st.session_state['authentication_status']:
        pages = [image_processing_page, account_management_page]
        if st.session_state['username'] == 'admin':
            pages.extend([debug_page, errors_page])
        pages.append(logout_page)
        pg = st.navigation(pages)

    else:
        pg = st.navigation(
            [login_page, registration_page],
        )

    return pg
