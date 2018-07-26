import torch
import torch.nn as nn
import numpy as np

class NER_Model(nn.Module):
    def __init__(self, blstm_hidden_size, num_layers, word_embedding_size, word_embedding_dim, feature_embedding_size, feature_embedding_dim, tag_size):
        super(NER_Model, self).__init__()
        self.num_layers = num_layers
        self.blstm_hidden_size = blstm_hidden_size
        self.word_embedding = nn.Embedding(num_embeddings=word_embedding_size, embedding_dim=word_embedding_dim)
        self.feature_embedding = nn.Embedding(num_embeddings=feature_embedding_size, embedding_dim=feature_embedding_dim)
        self.out = nn.LSTM((word_embedding_dim + feature_embedding_dim), blstm_hidden_size, bidirectional=True, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(blstm_hidden_size * 2, tag_size)
        self.softmax = nn.Softmax(dim=1)
        self.tag_size = tag_size
        self.transitions = nn.Parameter(
            torch.randn(self.tag_size, self.tag_size))
        self.START_TAG = 26
        self.STOP_TAG = 27

    def _get_lstm_features(self, sentences, features, length=None):
        embedd_word = self.word_embedding(sentences)
        embedd_feature = self.feature_embedding(features)
        embedd = torch.cat((embedd_word, embedd_feature), 2)
        if length is not None:
            embedd = torch.nn.utils.rnn.pack_padded_sequence(embedd, length, batch_first=True)
        init = self.init_hidden(sentences.size()[0])
        output, init = self.out(embedd, init)
        if length is not None:
            output = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0.0)
        logit = self.proj(output[0])

        return logit

    def init_hidden(self, batch_size):
        init_h = torch.ones(self.num_layers * 2, batch_size, self.blstm_hidden_size)
        init_c = torch.ones(self.num_layers * 2, batch_size, self.blstm_hidden_size)
        return (init_h, init_c)

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tag_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.START_TAG] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tag_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tag_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.STOP_TAG]
        alpha = self.log_sum_exp(terminal_var)
        return alpha


    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.START_TAG], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.STOP_TAG, tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tag_size), -10000.)
        init_vvars[0][self.START_TAG] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tag_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = self.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.STOP_TAG]
        best_tag_id = self.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.START_TAG  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, features, tags, length):
        loss = 0.
        feats = self._get_lstm_features(sentence, features, length)
        for i in range(sentence.size()[0]):
            forward_score = self._forward_alg(feats[i][:length[i]][:])
            gold_score = self._score_sentence(feats[i][:length[i]][:], tags[i][:length[i]])
            loss += forward_score - gold_score
        return loss

    def forward(self, sentence, features):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence, features)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def argmax(self, vec):
        # return the argmax as a python int
        _, idx = torch.max(vec, 1)
        return idx.item()

    def prepare_sequence(self, seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    # Compute log sum exp in a numerically stable way for the forward algorithm
    def log_sum_exp(self, vec):
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + \
               torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))