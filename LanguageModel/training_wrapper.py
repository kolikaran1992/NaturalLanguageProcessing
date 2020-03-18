from argparse import ArgumentParser
from pathlib import Path

from __logger__ import LOGGER_NAME
import logging
from LanguageModel.text_processor import TextProcessor
from __paths__ import path_to_lm, path_to_language_models
import json
from .__utils__ import token_extractor
from .model import LanguageModel

with open(path_to_lm.joinpath('params.json'), 'r') as f:
    params = json.load(f)

logger = logging.getLogger(LOGGER_NAME)


class TrainWrapper(object):
    def __init__(self,
                 model_name=None,
                 retrain=False,
                 texts_path=None
                 ):
        """
        :param model_name: name used for saving/loading language model
        :param retrain: if True model_name will be used to load language_model/text_processor
        :param texts_path: path of input text file
        """
        self._texts_path = texts_path
        self._model_name = model_name
        self._retrain = retrain

        if retrain:
            logger.info('Language model "{}" will be retrained from previous wts'.format(model_name))
        else:
            logger.info('Language model "{}" will be trained from randomly initialized wts'.format(model_name))

    def _get_text_processor(self):
        """
        --> return text processor
        --> load from existing if self._retrain
        :return: return text_processor
        """
        if not self._retrain:
            all_tokens = token_extractor(file_path=self._texts_path, sep='s#e#p#e#r#a#t#o#r',
                                         min_count=params.get('min_count'))

            processor = TextProcessor(max_seq_len=params.get('max_seq_len'),
                                      max_char_len=params.get('max_token_len'))
            processor.populate_vocab(all_tokens)
            processor.save(self._model_name)
        else:
            processor = TextProcessor()
            processor.load(self._model_name)

        return processor

    def _get_language_model(self, processor):
        """
        --> return language model
        --> load from existing if self._retrain
        :param processor: text_processor
        :return: language_model
        """
        if self._retrain:
            lang_model = LanguageModel()
            lang_model.load(self._model_name)
        else:
            lang_model = LanguageModel(
                word_embedding_size=params.get('word_emb_size'),
                char_embedding_size=params.get('char_emb_size'),
                word_inp_mask_val=processor._word_vocab.pad_idx(),
                word_vocab_size=len(processor._word_vocab),
                char_vocab_size=len(processor._char_vocab),
                max_seq_len=params.get('max_seq_len'),
                max_word_len=params.get('max_token_len'),
                char_cnn_filters=params.get('char_cnn_filters'),
                char_cnn_ker_size=params.get('char_cnn_ker_size'),
                char_cnn_pool_size=params.get('char_cnn_pool_size'),
                dropout=params.get('dropout')
            )
        return lang_model

    def get_model(self):
        """
        --> compile language model
        --> return model object of the language_model and text_processor
        :return: model, text_processor
        """
        text_processor = self._get_text_processor()
        language_model = self._get_language_model(text_processor)
        return language_model.get_model(), text_processor
