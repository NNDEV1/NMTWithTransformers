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

from decoder import Decoder, point_wise_feed_forward_network, build_encoder
from mhead_attention import MultiHeadAttention
from model_utils import *


class Config(object):
    def __init__(self, num_layers, d_model, dff, num_heads):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.dff = dff
        
        
class Transformer(tf.keras.Model):
    def __init__(self, config, target_vocab_size,
                 bert_config_file, bert_training=False,
                 rate=0.1, name="transformer"):
        super(Transformer, self).__init__(name=name)

        self.encoder = build_encoder(config_file=bert_config_file)
        self.encoder.trainable = bert_training
        self.decoder = Decoder(config.num_layers, config.d_model, config.num_heads,
                               config.dff, target_vocab_size, rate)
        
        self.final_linear = tf.keras.layers.Dense(target_vocab_size + 1)

    def load_stock_weights(self, bert, BertModelLayer, ckpt_file):
        assert isinstance(bert, BertModelLayer), "Expecting a BertModelLayer instance as first argument"
        assert tf.compat.v1.train.checkpoint_exists(ckpt_file), "Checkpoint does not exist: {}".format(ckpt_file)
        ckpt_reader = tf.train.load_checkpoint(ckpt_file)

        bert_prefix = 'transformer/bert'

        weights = []
        for weight in bert.weights:
            stock_name = map_to_stock_variable_name(weight.name, bert_prefix)
            if ckpt_reader.has_tensor(stock_name):
                value = ckpt_reader.get_tensor(stock_name)
                weights.append(value)
            else:
                raise ValueError("No value for:[{}], i.e.:[{}] in:[{}]".format(
                    weight.name, stock_name, ckpt_file))
        bert.set_weights(weights)
        print("Done loading {} BERT weights from: {} into {} (prefix:{})".format(
            len(weights), ckpt_file, bert, bert_prefix))
        
    def restore_encoder(self, bert_ckpt_file):

        self.load_stock_weights(self.encoder, bert_ckpt_file)

    def call(self, inp, tar, training, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training=self.encoder.trainable)

        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_linear(dec_output)

        return final_output, attention_weights


