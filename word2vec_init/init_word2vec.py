from gensim.models import Word2Vec
import os
import sys

path = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), '../data/train/sentence')
sentences = []
with open(path, 'r') as file:
    for line in file:
        print line.strip().split('\t')
        sentences.append(line.strip().split('\t'))

model = Word2Vec(sentences=sentences, window=5, min_count=1, max_vocab_size=10000, size=128, sg=1, iter=10)
model.wv.save_word2vec_format(fname=os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),'word2vec.txt'), fvocab=os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'vocab.txt'))
model.save(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),'model'))