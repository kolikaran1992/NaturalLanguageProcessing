from pathlib import Path

# root
root = Path(__file__).parent

# Language Model
path_to_lm = root.joinpath('LanguageModel')
path_to_language_models = path_to_lm.joinpath('saved_models')

# Vocabulary
path_to_saved_vocab = root.joinpath('vocabulary', 'saved_objects')