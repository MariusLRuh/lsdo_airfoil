__version__ = '0.1.0'

from pathlib import Path

_REPO_ROOT_FOLDER = Path(__file__).parents[0]
CONTROL_POINTS_FOLDER = _REPO_ROOT_FOLDER / 'core' / 'sample_airfoils' / 'control_points'
AIRFOIL_COORDINATES_FOLDER = _REPO_ROOT_FOLDER / 'core' / 'sample_airfoils' / 'raw_data'
MODELS_FOLDER = _REPO_ROOT_FOLDER / 'core' / 'models'
