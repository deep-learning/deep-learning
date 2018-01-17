import os
import tensorflow as tf
import numpy as np
import collections
import pickle as pkl
import re
import jieba
import os


class word2vec():
    def __init__(self,
                 vocab_list=None,
                 embedding_size=200,
                 window_len=3,
                 learning_rate=1,
                 num_sampled=100):
        assert type(vocab_list) is list
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.embedding_size = embedding_size
        self.window_len = window_len
        self.learning_rate = learning_rate
        self.num_sampled = num_sampled

        self.batch_size = 100

        # word -> id mapping
        self.word2id = {}
        for i in range(self.vocab_size):
            self.word2id[self.vocab_list[i]] = i

        self.train_words_num = 0
        self.train_sentences_num = 0
        self.train_times = 0

        self.build_graph()

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size], name='input')
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='output')

            self.embedding_dict = tf.Variable(tf.truncated_normal(shape=[self.vocab_size, self.embedding_size]))
            self.nec_weight = tf.Variable(tf.truncated_normal(shape=[self.vocab_size, self.embedding_size]))
            self.bias = tf.Variable(tf.zeros([self.vocab_size]))

            embed = tf.nn.embedding_lookup(self.embedding_dict,
                                           self.train_inputs)

            # define loss
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nec_weight,
                                                      biases=self.bias,
                                                      inputs=embed,
                                                      labels=self.train_labels,
                                                      num_sampled=self.num_sampled,
                                                      num_classes=self.vocab_size))
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

            self.test_word_id = tf.placeholder(tf.int32, shape=[None])
            # mo
            vec_l2_model = tf.sqrt(tf.reduce_sum(tf.square(self.embedding_dict, 1)))
            # gui yi hua
            self.normed_embedding = self.embedding_dict / vec_l2_model
            test_embed = tf.nn.embedding_lookup(self.embedding_dict, self.test_word_id)
            self.similarity = tf.matmul(test_embed, self.normed_embedding)

            self.init = tf.global_variables_initializer()

            self.saver = tf.train.Saver()

    def train_by_sentence(self, input_sentence=[]):
        sent_num = len(input_sentence)
        batch_inputs = []
        batch_labels = []
        for sent in input_sentence:
            for i in range(len(sent)):
                # [win_len, word, win_len]
                start = max(0, i - self.window_len)
                end = min(len(sent), i + self.window_len)
                for index in range(start, end):
                    if index == i:
                        continue
                    else:
                        input_id = self.word2id.get(sent[i])  # context
                        label_id = self.word2id.get(sent[index])
                        batch_inputs.append(input_id)
                        batch_labels.append(label_id)
        batch_inputs = np.array(batch_inputs, dtype=np.int32)
        batch_labels = np.array(batch_labels, dtype=np.int32)
        batch_labels = np.reshape(batch_labels, [len(batch_labels), 1])
        loss = self.sess.run(self.train_op, feed_dict={
            self.train_inputs: batch_inputs,
            self.train_labels: batch_labels
        })

        self.train_words_num += len(batch_inputs)
        self.train_sentences_num += len(input_sentence)

    def save_model(self, save_path):
        pass

    def load_model(self, model_path):
        pass


def load_stop_words(path):
    stop_words = []
    with open(path, encoding='utf-8') as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])  # strip \n
            line = f.readline()
    return set(stop_words)


def load_data(path):
    raw_word_list = []
    sentence_list = []
    with open(path, encoding='utf-8') as f:
        line = f.readline()
        while line:
            while '\n' in line:
                line = line.replace('\n', '')
            if len(line) > 0:
                raw_words = list(jieba.cut(line))
                dealed_words = []
                for word in raw_words:
                    if word not in stop_words:
                        raw_word_list.append(word)
                        dealed_words.append(word)
                sentence_list.append(dealed_words)
            line = f.readline()
    return raw_word_list, sentence_list


def count_freq(raw_words, most_common=3000):
    word_count = collections.Counter(raw_words).most_common(most_common)
    word_list = [x[0] for x in word_count]
    return word_list


if __name__ == '__main__':
    stop_words = load_stop_words('')
    raw_word_list, sentence_list = load_data('')
    word_list = count_freq(raw_word_list)
    w2v = word2vec(vocab_list=word_list,
                   embedding_size=200,
                   learning_rate=1,
                   num_sampled=100)
    nrof_steps = 10000
    for i in range(nrof_steps):
        sent = sentence_list[i]
        w2v.train_by_sentence([sent])
    w2v.save_model(save_path)
    w2v.load_model(model_path)
