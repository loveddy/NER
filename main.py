import torch
import sys
import os
import torch.optim as optim
import json
import numpy as np
from model import NER_Model
from gensim.models import Word2Vec
from utils import Utils

PRINT_EVERY = 5
SAVE_EVERY = 10
TEST_NUM = 256
BLSTM_HIDDEN = 256
WORD_EMBEDDING_DIM = 128
FEATURE_EMBEDDING_DIM = 8
NUM_LAYERS = 1
BATCH_SIZE = 32
EPOCH = 1000

def build_model(hidden_size, num_layers, word_embedding_dim, feature_embedding_dim):
    print("building model.......")
    with open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/vocab.json'), 'r') as file:
        word_dic = json.load(file)
    with open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/feature.json'), 'r') as file:
        feature_dic = json.load(file)
    with open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/tag.json'), 'r') as file:
        tag_dic = json.load(file)
    voc_size = len(word_dic)
    feature_size = len(feature_dic)
    tag_size = len(tag_dic)
    model = NER_Model(hidden_size, num_layers, voc_size, word_embedding_dim, feature_size, feature_embedding_dim, tag_size)
    if not os.path.exists(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/model')):
        word2vec = Word2Vec.load(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'word2vec_init/model'))
        trained = np.random.random(size=(voc_size, word_embedding_dim))
        trained[:-2][:] = np.array(word2vec.wv.vectors).reshape(voc_size - 2, word_embedding_dim)
        model.word_embedding.weight = torch.nn.Parameter(torch.tensor(trained, dtype=torch.float))
    else:
        model.load_state_dict(torch.load(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/model')))
    print("model is prepared........")
    return model, word_dic, feature_dic, tag_dic

def train():
    utils = Utils()
    utils.bulid_dic()
    utils.process_data()
    model, _, _, _ = build_model(BLSTM_HIDDEN, NUM_LAYERS, WORD_EMBEDDING_DIM, FEATURE_EMBEDDING_DIM)
    batch_size = BATCH_SIZE

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_list = []
    print("strat training.........")
    for i in range(EPOCH):
        loss = 0.
        optimizer.zero_grad()
        model.train()
        sen, f, tag, len_ = utils.get_random_batch(batch_size, mode='train')
        loss = model.neg_log_likelihood(sen,f, tag, len_)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item()/(batch_size * 1.))

        if (i+1) % PRINT_EVERY == 0:
            with torch.no_grad():
                model.eval()
                sen, f, tag, len_ = utils.get_random_batch(batch_size, mode='test')
                loss = model.neg_log_likelihood(sen, f, tag, len_)
                print("epoch:" + str(i+1) + " train loss: " + str(round(sum(loss_list[-PRINT_EVERY:])/(PRINT_EVERY * 1.0), 3)) + '  test loss:' + str(round(loss.item()/(batch_size * 1.0), 3)))
        if (i+1) % SAVE_EVERY ==0:
            torch.save(model.state_dict(), os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'ckpt/model'))

def test():
    model, _, _, _ = build_model(BLSTM_HIDDEN, NUM_LAYERS, WORD_EMBEDDING_DIM, FEATURE_EMBEDDING_DIM)
    utils = Utils()
    utils.bulid_dic()
    utils.process_data()
    with torch.no_grad():
        model.eval()
        count = 0
        all = 0
        for i in range(TEST_NUM):
            sen, f, tag, len_ = utils.get_random_batch(1, mode='test')
            score, predict = model.forward(sen, f)
            for j in range(len(predict)):
                if not tag[0][j] == 0 :
                    all +=1
                if predict[j] == tag[0][j] and not predict[j] == 0:
                    count += 1
        print("test accuracy:" + str(count / (all * 1.)))


if __name__ == '__main__':
    if sys.argv[1] == 'test':
        test()
    else:
        train()
