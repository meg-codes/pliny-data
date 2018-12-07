# Adapted from Tensorflow basic tutorial
import collections
import glob
import pickle
import re


from nltk.tokenize.punkt import PunktSentenceTokenizer
from cltk.stem.latin.j_v import JVReplacer
from cltk.stem.lemma import LemmaReplacer
import numpy as np
import tensorflow as tf


def combine_indices(a_list, indices):
    for index in indices:
        a_list[index:index+2] = [''.join(a_list[index:index+2])]
    return a_list


def load_pliny_corpus():
    """Loads the entire corpus of Pliny's letters
    (except book 10) as a string"""

    files = glob.glob('letters/[1]_*.txt')
    print(files[-1])
    total_strings = []
    for file in files:
        fp = open(file, 'r')
        input = fp.read()
        total_strings.append(input)
    corpus = '\n'.join(total_strings)
    lines = [line.strip() for line in corpus.split('\n') if line.strip()]
    for i, line in enumerate(lines):
        if line.endswith('-'):
            end = lines[i+1].split()[0]
            lines[i+1] = lines[i+1][len(end):]
            lines[i] = '%s%s' % (line[:-1], end)
        lines[i] = lines[i].strip()
    return ' '.join(lines)


def tokenize_corpus(corpus):
    tokenizer = PunktSentenceTokenizer()
    tokenizer.train(corpus)
    sentences = tokenizer.tokenize(corpus)
    return '\n'.join(sentences)


def de_que(word_list):

    QUE_COMPOUNDS = ['usque', 'que', 'isdemque', 'atque', 'quoque']

    que_indices = []
    for i, word in enumerate(word_list):
        if word.endswith('que'):
            if word not in QUE_COMPOUNDS:
                word_list[i] = word.split('que')[0]
                que_indices.append(i)
    shift = 1
    for i in que_indices:
        word_list.insert(i+shift, 'que')
        shift += 1
    return word_list


def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp


def main():
    corpus = tokenize_corpus(load_pliny_corpus())

    replacer = JVReplacer()
    corpus = replacer.replace(corpus)

    NO_PUNCT_RE = re.compile(r'[?!\.\'\"<>():;,]')

    sentences = [re.sub(NO_PUNCT_RE, '', sentence).lower() for sentence in corpus.split('\n')]

    # lemmatize the words

    lemmatizer = LemmaReplacer('latin')
    words_by_sentence = [lemmatizer.lemmatize(sent) for sent in sentences]
    all_words = [item for sublist in words_by_sentence for item in sublist]
    word_map = {}
    unk_count = 0

    count = [['UNK', -1]]
    count.extend(collections.Counter(all_words).most_common(299))
    for word, _ in count:
        word_map[word] = len(word_map)
    for word in all_words:
        index = word_map.get(word, 0)
        if index == 0:
            unk_count += 1
    count[0][1] = unk_count
    reverse_word_map = dict(zip(word_map.values(), word_map.keys()))

    data = []
    WINDOW_SIZE = 5

    for sentence in words_by_sentence:
        for i, word in enumerate(sentence):
            for nb_word in \
                sentence[max(i - WINDOW_SIZE, 0):min(i + WINDOW_SIZE, (len(sentence) + 1))]:
                if nb_word != word:
                    data.append([word_map.get(word, 0), word_map.get(nb_word, 0)])
    x_train = []
    y_train = []

    for word in data:
        x_train.append(to_one_hot(word[0], 300))
        y_train.append(to_one_hot(word[1], 300))

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    x = tf.placeholder(tf.float32, shape=(None, 300))
    y_label = tf.placeholder(tf.float32, shape=(None, 300))

    W1 = tf.Variable(tf.random_normal([300, 5]))
    b1 = tf.Variable(tf.random_normal([5]))

    hidden_representation = tf.add(tf.matmul(x, W1), b1)

    W2 = tf.Variable(tf.random_normal([5, 300]))
    b2 = tf.Variable(tf.random_normal([300]))
    prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_representation, W2), b2))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()

        sess.run(init) #make sure you do this!
        # define the loss function:
        cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
        # define the training step:
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)
        n_iters = 10000
        # train for n_iter iterations
        for _ in range(n_iters):

            start = np.random.randint(0, 700)
            end = start + 700
            if end > len(x_train) - 1:
                end = len(x_train) - 1
                start = len(x_train) - 701

            sess.run(train_step, feed_dict={x: x_train[start:end],
                                            y_label: y_train[start:end]})
            print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train[start:end], y_label: y_train[start:end]}))

        vectors = sess.run(W1 + b1)
        with open('word2int.pickle', 'wb') as fp:
            pickle.dump(word_map, fp)
        with open('int2word.pickle', 'wb') as fp:
            pickle.dump(reverse_word_map, fp)
        with open('vectors.pickle', 'wb') as fp:
            pickle.dump(vectors, fp)

if __name__ == '__main__':
    main()
