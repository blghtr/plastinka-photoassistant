import click
import subprocess
import sys
from pathlib import Path
from user_interface import get_app, setup_logging


# LLM:METADATA
# :hierarchy: [CLI | EntryPoint]
# :rationale: "Locate the main entry script for Streamlit execution."
# :contract: pre: "none", post: "returns Path object"
# LLM:END
def get_main_path():
    """Get the path to main.py, works both in development and after installation."""
    try:
        # Try to get path from importlib.resources (works after installation)
        import importlib.resources
        return importlib.resources.files(
            "plastinka_photoassistant"
        ).joinpath("main.py")
    except (ImportError, AttributeError):
        # Fallback to __file__ (works in development)
        return Path(__file__).resolve()

# LLM:METADATA
# :hierarchy: [CLI | Launcher]
# :relates-to: calls: "get_main_path", uses: "subprocess.run"
# :rationale: "Bootstrap the application via Streamlit CLI wrapper."
# :contract: pre: "streamlit installed", post: "application process started"
# LLM:END
def start():
    """Launch the Plastinka Photoassistant app with default configuration."""
    main_path = get_main_path()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(main_path),
        "--",
        "--config",
        "configs/pipeline/default_config.yaml"
    ]
    subprocess.run(cmd, check=True)


@click.command()
@click.option(
    '--config',
    default='configs/pipeline/default_config.yaml',
    help='Path to the pipeline config file.'
)
# LLM:METADATA
# :hierarchy: [CLI | Main]
# :relates-to: calls: "user_interface.setup_logging", calls: "user_interface.get_app"
# :rationale: "Initialize application context and run the main event loop."
# :contract: pre: "valid config path", post: "app running"
# LLM:END
def main(config):
    """Run the Plastinka Photoassistant app."""
    setup_logging()
    get_app(config_path=config).run()


if __name__ == '__main__':
    main()
