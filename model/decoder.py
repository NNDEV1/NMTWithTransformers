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

from model_utils import *
from mhead_attention import MultiHeadAttention

def point_wise_feed_forward_network(d_model, dff):

    return tf.keras.Sequential([
                tf.keras.layers.Dense(dff, activation='relu'),
                tf.keras.layers.Dense(d_model)
    ])

def build_encoder(config_file):
    with tf.io.gfile.GFile(config_file, "r") as reader:
        stock_params = StockBertConfig.from_json_string(reader.read())
        bert_params = stock_params.to_bert_model_layer_params()

    return BertModelLayer.from_params(bert_params, name="bert")
  
  
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate=rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=rate)
        self.dropout3 = tf.keras.layers.Dropout(rate=rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        
        att1, att_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        att1 = self.dropout1(att1, training=training)
        out1 = self.layernorm1(att1 + x)

        att2, att_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        att2 = self.dropout2(att2, training=training)
        out2 = self.layernorm2(att2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, att_weights_block1, att_weights_block2
      
      
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size + 1, d_model)
        self.pos_encoding = positional_encoding(target_vocab_size + 1, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, 
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, 
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2


        return x, attention_weights
