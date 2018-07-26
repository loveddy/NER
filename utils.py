#coding: UTF-8
import numpy as np
import os
import json
import torch
import sys
from tqdm import tqdm

class Utils(object):
    def __init__(self):
        self.dic = {}
        self.f_dic = {}
        self.tag_dic = {}
        self.train_sen_dic = {}
        self.train_f_dic = {}
        self.train_tag_dic = {}
        self.test_sen_dic = {}
        self.test_f_dic = {}
        self.test_tag_dic = {}


    def process_data(self):
        print("preparing data...............")
        if os.path.exists('data/train/sentence.json') and os.path.exists('data/train/features.json') and os.path.exists(
                'data/train/tag.json') and os.path.exists('data/test/sentence.json') and os.path.exists('data/test/features.json') and os.path.exists(
                'data/test/tag.json'):
            with open('data/train/sentence.json', 'r') as file:
                self.train_sen_dic = json.load(file)
            with open('data/train/features.json', 'r') as file:
                self.train_f_dic = json.load(file)
            with open('data/train/tag.json', 'r') as file:
                self.train_tag_dic = json.load(file)
            with open('data/test/sentence.json', 'r') as file:
                self.test_sen_dic = json.load(file)
            with open('data/test/features.json', 'r') as file:
                self.test_f_dic = json.load(file)
            with open('data/test/tag.json', 'r') as file:
                self.test_tag_dic = json.load(file)
        else:
            train_sentences = []
            train_features = []
            train_tags = []
            test_sentences = []
            test_features = []
            test_tags = []
            print("processing train data......")
            with open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'data/train/sentence'), 'r') as file:
                for line in file:
                    sen = line.decode('utf-8').strip().split('\t')
                    train_sentences.append(sen)
            with open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'data/train/feature'), 'r') as file:
                for line in file:
                    f = line.decode('utf-8').strip().split('\t')
                    train_features.append(f)
            with open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'data/train/tag'), 'r') as file:
                for line in file:
                    t = line.decode('utf-8').strip().split('\t')
                    train_tags.append(t)


            train_f2ids = self.f2id(train_features)
            train_tag2ids = self.tag2id(train_tags)
            train_sen2ids = self.sen2id(train_sentences)

            for line in train_sen2ids:
                self.train_sen_dic[len(self.train_sen_dic)] = line
            for line in train_f2ids:
                self.train_f_dic[len(self.train_f_dic)] = line
            for line in train_tag2ids:
                self.train_tag_dic[len(self.train_tag_dic)] = line



            print("processing test data......")

            with open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'data/test/sentence'), 'r') as file:
                for line in file:
                    sen = line.decode('utf-8').strip().split('\t')
                    test_sentences.append(sen)
            with open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'data/test/feature'), 'r') as file:
                for line in file:
                    f = line.decode('utf-8').strip().split('\t')
                    test_features.append(f)
            with open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'data/test/tag'), 'r') as file:
                for line in file:
                    t = line.decode('utf-8').strip().split('\t')
                    test_tags.append(t)
            test_sen2ids = self.sen2id(test_sentences)
            test_f2ids = self.f2id(test_features)
            test_tag2ids = self.tag2id(test_tags)
            for line in test_sen2ids:
                self.test_sen_dic[len(self.test_sen_dic)] = line
            for line in test_f2ids:
                self.test_f_dic[len(self.test_f_dic)] = line
            for line in test_tag2ids:
                self.test_tag_dic[len(self.test_tag_dic)] = line

            with open('data/train/sentence.json', 'w') as outfile:
                json.dump(self.train_tag_dic, outfile, ensure_ascii=False)
            with open('data/train/features.json', 'w') as outfile:
                json.dump(self.train_f_dic, outfile, ensure_ascii=False)
            with open('data/train/tag.json', 'w') as outfile:
                json.dump(self.train_tag_dic, outfile, ensure_ascii=False)
            with open('data/test/sentence.json', 'w') as outfile:
                json.dump(self.test_sen_dic, outfile, ensure_ascii=False)
            with open('data/test/features.json', 'w') as outfile:
                json.dump(self.test_f_dic, outfile, ensure_ascii=False)
            with open('data/test/tag.json', 'w') as outfile:
                json.dump(self.test_tag_dic, outfile, ensure_ascii=False)
        print("data prepared..........")



    def bulid_dic(self):
        print("building dic..........")
        if os.path.exists('ckpt/vocab.json') and os.path.exists('ckpt/feature.json') and os.path.exists('ckpt/tag.json'):
            with open('ckpt/vocab.json', 'r') as file:
                self.dic = json.load(file)
            with open('ckpt/feature.json', 'r') as file:
                self.f_dic = json.load(file)
            with open('ckpt/tag.json', 'r') as file:
                self.tag_dic = json.load(file)
        else:
            with open('word2vec_init/vocab.txt', 'r') as file:
                for line in file:
                    word, count = line.strip().split(' ')
                    if not word==None:
                        self.dic[word] = len(self.dic)
            self.dic['<PAD>'] = len(self.dic)
            self.dic['<UNK>'] = len(self.dic)
            with open('ckpt/vocab.json', 'w') as outfile:
                json.dump(self.dic, outfile, ensure_ascii=False)
            with open('ckpt/feature', 'r') as file:
                for line in file:
                    self.f_dic[line.strip()] = len(self.f_dic)
            with open('ckpt/tag', 'r') as file:
                for line in file:
                    self.tag_dic[line.strip()] = len(self.tag_dic)
            with open('ckpt/feature.json', 'w') as outfile:
                json.dump(self.f_dic, outfile, ensure_ascii=False)
            with open('ckpt/tag.json', 'w') as outfile:
                json.dump(self.tag_dic, outfile, ensure_ascii=False)

    def sen2id(self, sens):
        sen2ids = []
        for i in tqdm(range(len(sens))):
            temp = []
            for word in sens[i]:
                if word in self.dic.keys():
                    temp.append(self.dic[word])
                else:
                    temp.append(self.dic['<UNK>'])
            sen2ids.append(temp)
        return sen2ids

    def f2id(self, features):
        f2ids = []
        for i in tqdm(range(len(features))):
            temp = []
            for word in features[i]:
                temp.append(self.f_dic[word])
            f2ids.append(temp)
        return f2ids

    def tag2id(self, tags):
        tag2ids = []
        for i in tqdm(range(len(tags))):
            temp = []
            for word in tags[i]:
                temp.append(self.tag_dic[word])
            tag2ids.append(temp)
        return tag2ids

    def by_score(self, t):
        return len(t[0])

    def get_random_batch(self, batch_size, mode='train'):
        sample = []
        data = []

        if mode == 'train':
            for i in range(int(batch_size)):
                index = np.random.randint(0, len(self.train_sen_dic) - 1)
                while index in sample:
                    index = np.random.randint(0, len(self.train_sen_dic) - 1)
                sample.append(index)
                tt = []
                tt.append(self.train_sen_dic[str(index)])
                tt.append(self.train_f_dic[str(index)])
                tt.append(self.train_tag_dic[str(index)])
                data.append(tt)
        elif mode == 'test':
            for i in range(int(batch_size)):
                index = np.random.randint(0, len(self.train_sen_dic) - 1)
                while index in sample:
                    index = np.random.randint(0, len(self.train_sen_dic) - 1)
                sample.append(index)
                tt = []
                tt.append(self.train_sen_dic[str(index)])
                tt.append(self.train_f_dic[str(index)])
                tt.append(self.train_tag_dic[str(index)])
                data.append(tt)
        data_temp = sorted(data, key=self.by_score, reverse=True)
        batch_sen = []
        batch_f =[]
        batch_tag =[]
        len_ = []
        for line in data_temp:
            len_.append(len(line[0]))
            temp = line[0]
            for i in range(len(data_temp[0][0]) - len(line[0])):
                temp.append(self.dic['<PAD>'])

            batch_sen.append(temp)
            temp = line[1]
            for i in range(len(data_temp[0][1]) - len(line[1])):
                temp.append(self.f_dic['<pad>'])
            batch_f.append(temp)
            temp = line[2]
            for i in range(len(data_temp[0][2]) - len(line[2])):
                temp.append(self.tag_dic['<pad>'])
            batch_tag.append(temp)



        return torch.tensor(batch_sen, dtype=torch.long), torch.tensor(batch_f, dtype=torch.long), torch.tensor(batch_tag, dtype=torch.long), len_



