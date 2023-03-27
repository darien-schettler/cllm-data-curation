import os


def get_abs_current_file_path():
    """Return the absolute path of the project root directory."""
    return os.path.abspath(__file__)


def get_abs_pwd_path():
    """Return the absolute path of the present working directory."""
    return os.path.dirname(get_abs_current_file_path())


def get_abs_module_path():
    """Return the absolute path of the module directory."""
    return os.path.dirname(get_abs_pwd_path())


def get_abs_project_path():
    """Return the absolute path of the module directory."""
    return os.path.dirname(get_abs_module_path())