from nltk.tokenize import RegexpTokenizer

word_tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')
