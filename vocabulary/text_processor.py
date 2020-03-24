from sklearn.base import BaseEstimator, TransformerMixin
from .tokenizer import get_tokens


class Text2Toks(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, texts, y=None):
        return self

    def transform(self, texts, y=None):
        toks = [get_tokens(sent, tokens_only=False)[1] for sent in texts]
        return toks
