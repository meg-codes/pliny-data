import glob
import re

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from cltk.stem.latin.j_v import JVReplacer
from cltk.stem.lemma import LemmaReplacer


STOPS_LIST = []
with open('stopwords/stopwords_latin.txt') as fp:
    for line in fp:
        if not line.startswith('#'):
            STOPS_LIST.append(line.strip())


def import_pliny():

    letters = []
    files = glob.glob('letters/*.txt')
    for file in files:
        fp = open(file, 'r')
        input = fp.read()
        letters.append(((input), file.split('/')[-1]))
    return letters


def get_docs(letters):

    docs = []
    count = 0
    for i, entry in enumerate(letters):
        letter, tag = entry
        NO_PUNCT_RE = re.compile(r'[?!\.\'\"<>():;,]')
        replacer = JVReplacer()
        letter = replacer.replace(letter)
        words = re.sub(NO_PUNCT_RE, '', letter).lower().split()

        for i, word in enumerate(words):
            if word.endswith('-'):
                words[i+1] = '%s%s' % (word.strip('-'), words[i+1])
        words = [w for w in words if not w.endswith('-')]
        words = [w for w in words if w not in STOPS_LIST]
        words = ' '.join(words)
        lemmatizer = LemmaReplacer('latin')
        words = lemmatizer.lemmatize(words)
        count += len(words)
        doc = TaggedDocument(words, [tag])
        docs.append(doc)
    return docs


docs = get_docs(import_pliny())
model = Doc2Vec(docs, vector_size=100, window=10, min_count=1, workers=1)
model.train(docs, total_examples=len(docs), epochs=100)
model.save('pliny.model')
