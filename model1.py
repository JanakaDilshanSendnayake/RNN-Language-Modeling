import numpy as np
import collections
import torch
import torch.nn as nn
import random
from scipy.special import softmax

#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")

class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space.
    If it has occurred with more consonants than vowels, classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1

class RNNClassifier(ConsonantVowelClassifier, nn.Module):
    def __init__(self, dict_size, input_size, hidden_size, dropout, vocab_index):
        super(RNNClassifier, self).__init__()
        self.vocab_index = vocab_index
        self.word_embedding = nn.Embedding(dict_size, input_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Dropout = nn.Dropout(0.5)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, dropout=dropout)
        self.g = nn.ReLU()
        self.W = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.Softmax(dim=0)
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.rnn.weight_hh_l0, a=-1.0/np.sqrt(self.hidden_size), b=1.0/np.sqrt(self.hidden_size))
        nn.init.uniform_(self.rnn.weight_ih_l0, a=-1.0 / np.sqrt(self.hidden_size), b=1.0 / np.sqrt(self.hidden_size))
        nn.init.uniform_(self.rnn.bias_hh_l0, a=-1.0 / np.sqrt(self.hidden_size), b=1.0 / np.sqrt(self.hidden_size))
        nn.init.uniform_(self.rnn.bias_ih_l0, a=-1.0 / np.sqrt(self.hidden_size), b=1.0 / np.sqrt(self.hidden_size))
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, context):
        embedded_input = self.word_embedding(context)
        embedded_input = embedded_input.unsqueeze(1)
        init_state = (torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float(),
                      torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float())
        output, (hidden_state, cell_state) = self.rnn(embedded_input, init_state)
        zz = hidden_state.squeeze()
        zz1 = self.W(self.Dropout(self.g(zz)))
        zz1 = self.log_softmax(zz1)
        return zz1

    def predict(self, context):
        x1 = [self.vocab_index.index_of(ex) for ex in context]
        x = form_input(x1)
        log_probs = self.forward(x)
        res = torch.argmax(log_probs).item()
        return res

def form_input(x) -> torch.Tensor:
    """
    Form the input to the neural network.
    :param x: a [num_samples x inp] numpy array containing input data
    :return: a [num_samples x inp] Tensor
    """
    return torch.from_numpy(np.asarray(x)).int()

def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)

def prepare_data(train_cons_exs, train_vowel_exs):
    train_exs = []
    for ex in train_cons_exs:
        train_exs.append([ex, 0])
    for ex in train_vowel_exs:
        train_exs.append([ex, 1])
    return train_exs

def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    # define hyperparameters
    batch_size = 1
    dict_size = 27
    input_size = 20
    hidden_size = 50
    dropout = 0.5
    num_epochs = 5
    num_classes = 2
    learning_rate = 0.001
    criterion = nn.BCELoss()

    train_exs = prepare_data(train_cons_exs, train_vowel_exs)
    rnn = RNNClassifier(dict_size, input_size, hidden_size, dropout, vocab_index)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    ex_idx = [i for i in range(0, len(train_exs))]

    for epoch in range(0, num_epochs):
        rnn.train()
        ex_indices = ex_idx.copy()
        random.shuffle(ex_indices)
        total_loss = 0.0

        for i, idx in enumerate(ex_indices, 1):
            x1 = [vocab_index.index_of(ex) for ex in train_exs[idx][0]]
            x = form_input(x1)
            y = train_exs[idx][1]
            y_onehot = torch.zeros(num_classes)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)

            rnn.zero_grad()
            log_probs = rnn.forward(x)
            loss = criterion(log_probs, y_onehot)
            total_loss += loss

            if i % 1000 == 0:
                print(f"Epoch loss on epoch {epoch}, {i}: {loss}")

            loss.backward()
            optimizer.step()

        print(f"Total loss on epoch {epoch}: {total_loss}")

        rnn.eval()
        num_correct = 0
        for ex in train_cons_exs:
            if rnn.predict(ex) == 0:
                num_correct += 1
        for ex in train_vowel_exs:
            if rnn.predict(ex) == 1:
                num_correct += 1
        num_total = len(train_cons_exs) + len(train_vowel_exs)
        train_accuracy = float(num_correct) / num_total * 100.0
        print("=====Results=====")
        print(f"train data accuracy: {train_accuracy:.2f}%")

        num_correct = 0
        for ex in dev_cons_exs:
            if rnn.predict(ex) == 0:
                num_correct += 1
        for ex in dev_vowel_exs:
            if rnn.predict(ex) == 1:
                num_correct += 1
        num_total = len(dev_cons_exs) + len(dev_vowel_exs)
        dev_accuracy = float(num_correct) / num_total * 100.0
        print("=====Results=====")
        print(f"dev data accuracy: {dev_accuracy:.2f}%")

    return rnn


class LanguageModel(object):
    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")

class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)

class RNNLanguageModel(LanguageModel, nn.Module):
    def __init__(self, dict_size, input_size, hidden_size, dropout, vocab_index):
        super(RNNLanguageModel, self).__init__()
        self.vocab_index = vocab_index
        self.word_embedding = nn.Embedding(dict_size, input_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, dropout=dropout)
        self.g = nn.ReLU()
        self.V = nn.Linear(hidden_size, 512)
        self.W = nn.Linear(512, 27)
        self.softmax = nn.Softmax(dim=0)
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.lstm.weight_hh_l0, a=-1.0/np.sqrt(self.hidden_size), b=1.0/np.sqrt(self.hidden_size))
        nn.init.uniform_(self.lstm.weight_ih_l0, a=-1.0/np.sqrt(self.hidden_size), b=1.0/np.sqrt(self.hidden_size))
        nn.init.uniform_(self.lstm.bias_hh_l0, a=-1.0/np.sqrt(self.hidden_size), b=1.0/np.sqrt(self.hidden_size))
        nn.init.uniform_(self.lstm.bias_ih_l0, a=-1.0/np.sqrt(self.hidden_size), b=1.0/np.sqrt(self.hidden_size))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.V.weight)

    def forward(self, context):
        embedded_input = self.word_embedding(context)
        embedded_input = embedded_input.unsqueeze(1)
        init_state = (torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float(),
                      torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float())
        output, (hidden_state, cell_state) = self.lstm(embedded_input, init_state)
        output = output.squeeze()
        hidden_state = hidden_state.squeeze()
        out = self.W(self.Dropout(self.g(self.V(self.Dropout(self.g(output))))))
        hidden = self.W(self.Dropout(self.g(self.V(self.Dropout(self.g(hidden_state))))))
        return out, hidden

    def get_next_char_log_probs(self, context):
        x1 = [self.vocab_index.index_of(ex) for ex in context]
        x = form_input(x1)
        out, hidden = self.forward(x)
        hidden1 = hidden.detach().numpy()
        hidden2 = softmax(hidden1)
        hidden3 = np.log(hidden2)
        return hidden3

    def get_log_prob_sequence(self, next_chars, context):
        log_probs = 0.0
        for i in range(len(next_chars)):
            next_char_log_probs = self.get_next_char_log_probs(context + next_chars[0:i])
            log_probs += next_char_log_probs[self.vocab_index.index_of(next_chars[i])]
        return log_probs


def read_chunk_data(train_text, dev_text, input_size):
    train_exs = []
    dev_exs = []

    # Process train text
    for i in range(0, len(train_text), input_size):
        chunk = train_text[i:i + input_size]
        if len(chunk) < input_size:
            chunk = chunk.ljust(input_size)  # Pad with spaces if needed
        train_exs.append(chunk)

    # Process dev text
    for i in range(0, len(dev_text), input_size):
        chunk = dev_text[i:i + input_size]
        if len(chunk) < input_size:
            chunk = chunk.ljust(input_size)  # Pad with spaces if needed
        dev_exs.append(chunk)

    return train_exs, dev_exs


# The rest of your code remains the same...

def train_lm(args, train_text, dev_text, vocab_index):
    # define hyperparameters
    batch_size = 1
    dict_size = 27
    input_size = 20
    hidden_size = 50
    dropout = 0.5
    num_epochs = 9
    num_classes = 27
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()

    # process training data and label:
    train_exs, dev_exs = read_chunk_data(train_text, dev_text, input_size)
    rnnlm = RNNLanguageModel(dict_size, input_size, hidden_size, dropout, vocab_index)
    optimizer = torch.optim.Adam(rnnlm.parameters(), lr=learning_rate)
    ex_idx = list(range(len(train_exs)))

    for epoch in range(num_epochs):
        rnnlm.train()
        ex_indices = ex_idx.copy()
        random.shuffle(ex_indices)
        total_loss = 0.0

        for i, idx in enumerate(ex_indices, 1):
            out_seq = [vocab_index.index_of(ex) for ex in train_exs[idx]]
            input_seq = [vocab_index.index_of(' ')] + out_seq[:-1]
            x = form_input(input_seq)
            y = torch.tensor(np.asanyarray(out_seq), dtype=torch.long)

            rnnlm.zero_grad()
            out, _ = rnnlm.forward(x)
            loss = criterion(out, y)
            total_loss += loss

            if i % 1000 == 0:
                print(f"Epoch loss on epoch {epoch + 1}, {i}: {loss:.4f}")

            loss.backward()
            optimizer.step()

        print(f"Total training loss on epoch {epoch + 1}: {total_loss:.4f}")

        # Evaluate on the dev set
        rnnlm.eval()
        total_loss = 0.0

        for i, idx in enumerate(range(len(dev_exs)), 1):
            out_seq = [vocab_index.index_of(ex) for ex in dev_exs[idx]]
            input_seq = [vocab_index.index_of(' ')] + out_seq[:-1]
            x = form_input(input_seq)
            y = torch.tensor(np.asanyarray(out_seq), dtype=torch.long)

            out, _ = rnnlm.forward(x)
            loss = criterion(out, y)
            total_loss += loss

            if i % 1000 == 0:
                print(f"Dev loss on epoch {epoch + 1}, {i}: {loss:.4f}")

        print(f"Total dev loss on epoch {epoch + 1}: {total_loss:.4f}")

    return rnnlm