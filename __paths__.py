from pathlib import Path

root = Path(__file__).parent
path_to_lm = root.joinpath('LanguageModel')
path_to_logs = path_to_lm.joinpath('logs')
path_to_models = path_to_lm.joinpath('saved_models')
