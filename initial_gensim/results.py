
from gensim.models import Word2Vec
from gensim.models.ldamodel import LdaModel


model = LdaModel.load('pliny_lda.model')
topics = model.print_topics(num_words=4)
for t in topics:
    print(t)
