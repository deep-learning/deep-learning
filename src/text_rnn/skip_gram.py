import tensorflow as tf
import numpy as np
import os
import math

batch_size = 64
embedding_dimension = 5
negative_samples = 8
LOG_DIR = 'logs/word2vec'

digit_to_word_map = {
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine'
}

sentences = []

for i in range(10000):
    rand_odd_ints = np.random.choice(range(1, 10, 2), 3)
    sentences.append(' '.join([digit_to_word_map[r] for r in rand_odd_ints]))
    rand_even_ints = np.random.choice(range(2, 10, 2), 3)
    sentences.append(' '.join([digit_to_word_map[r] for r in rand_even_ints]))

print(sentences[0:10])


def build_mapping(sentences):
    word2index_map = {}
    index = 0
    for sent in sentences:
        for word in sent.lower().split():
            if word not in word2index_map:
                word2index_map[word] = index
                index += 1
    index2word_map = {index: word for word, index in word2index_map.items()}
    return word2index_map, index2word_map


word2index_map, index2word_map = build_mapping(sentences)
vocabulary_size = len(index2word_map)
print(word2index_map)
print(index2word_map)
print(vocabulary_size)

# generate skip-gram pairs
skip_gram_pairs = []
for sent in sentences:
    tokenized_sent = sent.lower().split()
    for i in range(1, len(tokenized_sent) - 1):
        # [[before, after], current]
        word_context_pair = [[word2index_map[tokenized_sent[i - 1]],
                              word2index_map[tokenized_sent[i + 1]]],
                             word2index_map[tokenized_sent[i]]]
        # [current, before]
        skip_gram_pairs.append([word_context_pair[1], word_context_pair[0][0]])
        # [current, after]
        skip_gram_pairs.append([word_context_pair[1], word_context_pair[0][1]])

print(skip_gram_pairs[:10])


def get_skip_gram_batch(batch_size):
    instance_indices = list(range(len(skip_gram_pairs)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [skip_gram_pairs[i][0] for i in batch]
    y = [[skip_gram_pairs[i][1]] for i in batch]
    return x, y


x_batch, y_batch = get_skip_gram_batch(8)
print(x_batch)
print(y_batch)

# train
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

with tf.name_scope('embeddings'):
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_dimension], minval=-1.0, maxval=1.0),
                             name='embedding')
    # a lookup table
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

loss =

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(learning_rate=0.1,
                                global_step=global_step,
                                decay_steps=1000,
                                decay_rate=0.95,
                                staircase=True)
train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
