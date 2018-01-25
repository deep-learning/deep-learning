import numpy as np
import tensorflow as tf

batch_size = 128
embedding_dimension = 64
num_classes = 2
hidden_layer_size = 32
times_steps = 6
element_size = 1

digit_to_word_map = {
    0: 'pad',
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: 'seven',
    8: 'eight',
    9: 'nine'
}

even_sentences = []
odd_sentences = []
seqlens = []

for i in range(10000):
    rand_seq_len = np.random.choice(range(3, 7))
    seqlens.append(rand_seq_len)
    rand_odd_ints = np.random.choice(range(1, 10, 2),
                                     rand_seq_len)
    rand_even_ints = np.random.choice(range(2, 10, 2),
                                      rand_seq_len)

    # padding
    if rand_seq_len < 6:
        rand_odd_ints = np.append(rand_odd_ints,
                                  [0] * (6 - rand_seq_len))
        rand_even_ints = np.append(rand_even_ints,
                                   [0] * (6 - rand_seq_len))

    even_sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))
    odd_sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))
data = even_sentences + odd_sentences

# same seq lengths for even, odd sentences
seqlens *= 2

# map from words to indices
word2index_map = {}
index = 0
for sent in data:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1

# inverse map
index2word_map = {index: word for word, index in word2index_map.items()}

vocabulary_size = len(index2word_map)

labels = [1] * 10000 + [0] * 10000

for i in range(len(labels)):
    label = labels[i]
    one_hot_encoding = [0] * 2
    one_hot_encoding[label] = 1
    labels[i] = one_hot_encoding

data_indices = list(range(len(data)))
np.random.shuffle(data_indices)
data = np.array(data)[data_indices]
