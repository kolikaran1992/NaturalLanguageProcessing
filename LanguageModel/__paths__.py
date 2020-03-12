from pathlib import Path

root = Path(__file__).parent
path_to_logs = root.joinpath('logs')
path_to_models = root.joinpath('saved_models')