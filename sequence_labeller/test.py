from LanguageModel.model import LanguageModel

lm = LanguageModel()
lm.load('work_local')


lm.get_model().layers[-1]