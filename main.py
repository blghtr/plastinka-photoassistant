import click
import subprocess
import sys
from pathlib import Path
from user_interface import get_app


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
        "configs/default_config.yaml"
    ]
    subprocess.run(cmd, check=True)


@click.command()
@click.option(
    '--config',
    default='configs/default_config.yaml',
    help='Path to the pipeline config file.'
)
def main(config):
    """Run the Plastinka Photoassistant app."""
    get_app(config_path=config).run()


if __name__ == '__main__':
    main()
