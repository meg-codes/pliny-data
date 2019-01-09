import glob
import re


from gensim.models import Word2Vec
from nltk.tokenize.punkt import PunktSentenceTokenizer
from cltk.stop.stop import CorpusStoplist
from cltk.stem.latin.j_v import JVReplacer
from cltk.stem.lemma import LemmaReplacer

STOPS_LIST = []
with open('../stopwords/stopwords_latin.txt') as fp:
    for line in fp:
        if not line.startswith('#'):
            STOPS_LIST.append(line.strip())

def load_pliny_corpus():
    """Loads the entire corpus of Pliny's letters
    (except book 10) as a string"""

    files = glob.glob('../letters/*.txt')
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


def main():

    corpus = tokenize_corpus(load_pliny_corpus())

    replacer = JVReplacer()
    corpus = replacer.replace(corpus)

    NO_PUNCT_RE = re.compile(r'[?!\.\'\"<>():;,]')

    sentences = [re.sub(NO_PUNCT_RE, '', sentence).lower() for sentence in corpus.split('\n')]
    # lemmatize the words
    lemmatizer = LemmaReplacer('latin')
    for i, val in enumerate(sentences):
        sentences[i] = [w.lower() for w in sentences[i].split() if w.lower() not in STOPS_LIST]
    words_by_sentence = [lemmatizer.lemmatize(sent) for sent in sentences]
    for i, val in enumerate(words_by_sentence):
        words_by_sentence[i] = [w for w in words_by_sentence[i] if w not in
        ['qui1', 'edo', 'sum', 'a', 'se', 'quod', 'me'] ]

    model = Word2Vec(
        words_by_sentence,
        size=150,
        window=10,
        min_count=2,
    )
    model.train(words_by_sentence, total_examples=len(sentences), epochs=10000)

    model.save("pliny.model")

if __name__ == '__main__':
    main()
