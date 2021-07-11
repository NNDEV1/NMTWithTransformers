from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import time
import numpy as np
import matplotlib.pyplot as plt
import collections
import unicodedata
import os
from bert import BertModelLayer
from bert.loader import StockBertConfig, load_stock_weights
from bert import bert_tokenization
from bert.loader import map_to_stock_variable_name

from train import transformer, create_masks
from dataset import tokenizer_zh, tokenizer_en, max_seq_length

def encode_zh(zh):
    tokens_zh = tokenizer_zh.tokenize(zh)
    lang1 = tokenizer_zh.convert_tokens_to_ids(['[CLS]'] + tokens_zh + ['[SEP]'])

    return lang1

def evaluate(transformer, inp_sentence):
    # normalize input sentence
    inp_sentence = encode_zh(inp_sentence)
    encoder_input = tf.expand_dims(inp_sentence, 0)

    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(max_seq_length):
        combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     combined_mask,
                                                     dec_padding_mask)

        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, tokenizer_en.vocab_size + 1):
            return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights
  

def translate(transformer, sentence):
  
  result, attention_weights = evaluate(transformer, sentence)

  predicted_sentence = tokenizer_en.decode([i for i in result
                                            if i < tokenizer_en.vocab_size])

  print('Input: {}'.format(sentence))
  print('Predicted translation: {}'.format(predicted_sentence))
