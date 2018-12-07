import glob
import os
import string

# import lemmatizers and Latin specific models and utilities
from cltk.stem.lemma import LemmaReplacer
from cltk.stem.latin.j_v import JVReplacer
from cltk.corpus.utils.importer import CorpusImporter
from cltk.stop.latin.stops import STOPS_LIST

SOURCE_DIR = 'letters'
OUTPUT_DIR = 'letters_lemmatized'


# import corpus and model if not already imported
corpus_importer = CorpusImporter('latin')
corpus_importer.import_corpus('latin_models_cltk')


# instantiate a lemmatizer and the jv replacer in Latin (we need ui)
lemmatizer = LemmaReplacer('latin')
replacer = JVReplacer()

if not os.path.isdir('letters_lemmatized'):
    os.mkdir('letters_lemmatized')

source_files = glob.glob('%s/*.txt' % SOURCE_DIR)

for filepath in source_files:
    all_tokens = []
    with open(filepath, 'r') as fp:
        for line in fp:
            # Remove punctuation as accurately as possible
            table = str.maketrans({key: None for key in string.punctuation})
            line = line.translate(table)
            tokens = []
            tokens = replacer.replace(line.lower())
            tokens = lemmatizer.lemmatize(tokens)

            all_tokens += [token for token in tokens
                           if token not in STOPS_LIST]
        with open(os.path.join(OUTPUT_DIR, filepath.split('/')[1]), 'w') as outfile:
                outfile.write(' '.join(all_tokens))
