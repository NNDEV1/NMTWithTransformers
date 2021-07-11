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

config = tfds.translate.wmt.WmtConfig(
    description="WMT 2019 translation task dataset.",
    version="0.0.3",
    language_pair=("zh", "en"),
    subsets={
        tfds.Split.TRAIN: ["newscommentary_v13"],
        tfds.Split.VALIDATION: ["newsdev2017"],
    }
)

builder = tfds.builder("wmt_translate", config=config)
print(builder.info.splits)
builder.download_and_prepare()
datasets = builder.as_dataset(as_supervised=True)
print('datasets is {}'.format(datasets))

train_examples = datasets['train']
validation_examples = datasets['validation']

vocab_file = "vocab_en"
if os.path.isfile(vocab_file + '.subwords'):
    tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.load_from_file(vocab_file)

else:
    tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for zh, en in train_examples), target_vocab_size=2 ** 13)
    tokenizer_en.save_to_file('vocab_en')
    
    
tokenizer_zh = bert_tokenization.FullTokenizer(vocab_file='/content/chinese_L-12_H-768_A-12/vocab.txt', do_lower_case=True)

max_seq_length = 128

def encode(zh, en, seq_length=max_seq_length):
    tokens_zh = tokenizer_zh.tokenize(tf.compat.as_text(zh.numpy()))
    lang1 = tokenizer_zh.convert_tokens_to_ids(['[CLS]'] + tokens_zh + ['[SEP]'])
    if len(lang1) < seq_length:
      lang1 = lang1 + list(np.zeros(seq_length - len(lang1), 'int32'))
      
    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        tf.compat.as_text(en.numpy())) + [tokenizer_en.vocab_size + 1]
    if len(lang2) < seq_length:
      lang2 = lang2 + list(np.zeros(seq_length - len(lang2), 'int32'))
      
    return lang1, lang2
  
def filter_max_length(x, y, max_length=max_seq_length):
  return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)

BUFFER_SIZE = 50000
BATCH_SIZE = 64

train_dataset = train_examples.map(
    lambda zh, en: tf.py_function(encode, [zh, en], [tf.int32, tf.int32])
)

train_dataset = train_dataset.filter(filter_max_length)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
    BATCH_SIZE, padded_shapes=([-1], [-1]), drop_remainder=True
)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

validation_examples = validation_examples.map(
    lambda zh, en: tf.py_function(encode, [zh, en], [tf.int32, tf.int32]))
validation_examples = validation_examples.filter(filter_max_length)
validation_examples = validation_examples.padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))

