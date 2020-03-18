from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D, MaxPool1D, \
    Flatten, concatenate
from __layers__ import *
from __logger__ import LOGGER_NAME
import logging
from __paths__ import path_to_lm, path_to_language_models
import json

logger = logging.getLogger(LOGGER_NAME)

with open(path_to_lm.joinpath('params.json'), 'r') as f:
    params = json.load(f)


class LanguageModel(object):
    def __init__(self,
                 word_embedding_size=None,
                 char_embedding_size=None,
                 word_inp_mask_val=None,
                 word_vocab_size=None,
                 char_vocab_size=None,
                 max_seq_len=None,
                 max_word_len=None,
                 char_cnn_filters=None,
                 char_cnn_ker_size=None,
                 char_cnn_pool_size=None,
                 dropout=0.0
                 ):
        self._model = None
        self._max_seq_len = max_seq_len
        self._max_word_len = max_word_len
        self._word_emb_size = word_embedding_size
        self._char_emb_size = char_embedding_size
        self._word_inp_mask = word_inp_mask_val
        self._word_vocab_size = word_vocab_size
        self._char_vocab_size = char_vocab_size
        self._char_cnn_filters = char_cnn_filters
        self._char_cnn_ker_size = char_cnn_ker_size
        self._char_cnn_pool_size = char_cnn_pool_size
        self._dropout = dropout

    def _word_embedding_layer(self, word_in, train_wts=True):
        """
        --> get word embeddings
        :param word_in: tensor of word idxs (2d)
        :param train_wts: (boolean) if True then wts will be trained
        :return: a sequence of word vectors
        """
        masked_word_in = Mask(mask_value=self._word_inp_mask, name='masked_word_inputs')(word_in)
        emb_word = Embedding(input_dim=self._word_vocab_size,
                             output_dim=self._word_emb_size,
                             input_length=self._max_seq_len,
                             name='word_embeddings', trainable=train_wts)(masked_word_in)
        return emb_word

    def _char_embedding_layer(self, char_in, train_wts=True):
        """
        --> get character level encodings
        :param char_in: a tensor of char idxs (3d)
        :param train_wts: (boolean) if True then wts will be trained
        :return: a sequence of character encodings
        """
        emb_char = TimeDistributed(Embedding(input_dim=self._char_vocab_size, output_dim=self._char_emb_size,
                                             input_length=self._max_word_len,
                                             name='char_embeddings'),
                                   name='time_distributed_char_emb',
                                   trainable=train_wts)(char_in)
        emb_char = TimeDistributed(Conv1D(filters=self._char_cnn_filters,
                                          kernel_size=self._char_cnn_ker_size, activation='relu', name='char_cnn',
                                          trainable=train_wts),
                                   name='time_distributed_char_cnn',
                                   trainable=train_wts)(emb_char)
        emb_char = TimeDistributed(Dropout(self._dropout), name='timedistributed_char_cnn_dropout')(emb_char)
        emb_char = TimeDistributed(MaxPool1D(pool_size=self._char_cnn_pool_size), name='char_cnn_pooling')(emb_char)
        emb_char = TimeDistributed(Flatten(name='char_cnn_flatten'), name='time_dist_char_cnn_flatten')(emb_char)

        return emb_char

    def _embedding_layer(self, word_in, char_in, train_wts=True):
        """
        --> final embedding layer
        --> merge all embeddings
        :param word_in: tensor of word idxs (2d)
        :param char_in: tensor of char idxs (3d)
        :param train_wts: (boolean) if True then wts will be trained
        :return: final merged embeddings
        """
        word_emb = self._word_embedding_layer(word_in, train_wts=train_wts)
        char_emb = self._char_embedding_layer(char_in, train_wts=train_wts)

        merged_embedding = concatenate([word_emb, char_emb], name='merged_embeddings')
        final_emb = PosEmb(self._max_seq_len, merged_embedding.shape[-1], name='final_embedding')(merged_embedding)

        return final_emb

    def _elmo_lstm_out(self, inp, name='', rev=False, train_wts=True):
        """
        --> elmo style architecture
        --> input is fed to 1st rnn
        --> input merged with the output of first rnn is fed to the second rnn
        :param inp: (tensor) merged embeddings
        :param name: (str) name to be given to layers
        :param rev: (boolean) rnns will be reversed if True
        :param train_wts: (boolean) if True then wts will be trained
        :return: tuple of tensors (lstm1_out, lstm2_out)
        """
        _lstm1 = LSTM(units=inp.get_shape().as_list()[-1] // 2,
                      return_sequences=True,
                      recurrent_dropout=self._dropout, go_backwards=rev,
                      name='{}_1'.format(name), trainable=train_wts)(inp)
        merged_inp = concatenate([inp, _lstm1], name='{}_merge_lstm1_embed'.format(name))
        _lstm2 = LSTM(units=inp.get_shape().as_list()[-1] // 2,
                      return_sequences=True,
                      recurrent_dropout=self._dropout,
                      name='{}_2'.format(name), trainable=train_wts)(merged_inp)
        return _lstm1, _lstm2

    def get_model(self):
        """
        --> build model
        :return: self._model
        """
        if self._model:
            return self._model

        word_in = Input(shape=(self._max_seq_len,), name='token_inputs', dtype=params.get('dtype_int'))
        char_in = Input(shape=(self._max_seq_len, self._max_word_len,), name='char_inputs',
                        dtype=params.get('dtype_int'))

        embedding_layer = self._embedding_layer(word_in, char_in, train_wts=True)

        _, l2r_lstm = self._elmo_lstm_out(embedding_layer, name='l2r_lstm', rev=False, train_wts=True)
        _, r2l_lstm = self._elmo_lstm_out(embedding_layer, name='r2l_lstm', rev=True, train_wts=True)

        # merge left2right and right2left lstm outs
        merged = concatenate([l2r_lstm, r2l_lstm], name='merged_lstm1_lstm2')

        model = TimeDistributed(Dense(self._word_vocab_size, name='token_prediction', activation='softmax'),
                                name='time_dist_token_pred_layer')(merged)
        model = Model(inputs=[word_in, char_in], outputs=[model], name='LanguageModel')
        self._model = model
        return self._model

    def save(self, name):
        """
        --> saves model params to "LanguageModel/saved_models/name/model_params.json"
        --> saves model wts to "LanguageModel/saved_models/name/wts.h5"
        :param name (str): name for saving the model
        :return: None
        """
        path_to_vocab = path_to_language_models.joinpath(name)
        path_to_vocab.mkdir(parents=True, exist_ok=True)

        with open(path_to_vocab.joinpath("model_params.json"), 'w', encoding='utf-8') as f:
            all_items = {k: v for k, v in self.__dict__.items() if k not in ['_model']}
            json.dump(all_items, f)

        self._model.save_weights(path_to_vocab.joinpath('wts.h5'))

        logger.info('language model saved successfully at {}'.format(path_to_vocab))

    def load(self, name):
        """
        --> loads model params from "LanguageModel/saved_models/name/model_params.json"
        --> loads model wts from "LanguageModel/saved_models/name/wts.h5"
        :param name (str): name for loading the model
        :return: None
        """
        path_to_vocab = path_to_language_models.joinpath(name)
        if not path_to_vocab.is_dir():
            logger.error('language model could not be loaded since "" is not a valid directory'.format(path_to_vocab))
            exit(1)

        if not path_to_vocab.joinpath("model_params.json").is_file() or not path_to_vocab.joinpath('wts.h5').is_file():
            logger.error(
                'language model could not be loaded since params/wts does not exists at {}'.format(path_to_vocab))

        with open(path_to_vocab.joinpath("model_params.json"), 'r', encoding='utf-8') as f:
            obj = json.load(f)
            for k, v in obj.items():
                self.__setattr__(k, v)

        # build model
        self.get_model()

        wts_path = sorted(path_to_vocab.glob('*.h5'), key=lambda p: p.lstat().st_mtime)[-1]
        self._model.load_weights(wts_path)
        logger.info('name of loaded wts for language model {} = {}'.format(name, wts_path.name))

        logger.info('language model loaded successfully from {}'.format(path_to_vocab))
