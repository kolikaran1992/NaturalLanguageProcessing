from nltk.tokenize import RegexpTokenizer

word_tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')


def get_tokens(string, lower=False):
    tokens = word_tokenizer.tokenize(string.lower() if lower else string)
    return tokens
