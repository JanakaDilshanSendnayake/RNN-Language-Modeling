# lm.py

import argparse
import time
from models import *
from utils import *

####################################################
# DO NOT MODIFY THIS FILE IN YOUR FINAL SUBMISSION #
####################################################


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='lm.py')
    parser.add_argument('--model', type=str, default='UNIFORM', help='model to run (UNIFORM or RNN)')
    parser.add_argument('--train_path', type=str, default='data/text8-100k.txt', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/text8-dev.txt', help='path to dev set (you should not need to modify)')
    args = parser.parse_args()
    return args


def read_text(file):
    """
    :param file:
    :return: The text in the given file as a single string
    """
    all_text = ""
    for line in open(file):
        all_text += line.strip()
    print("%i chars read in" % len(all_text))
    return all_text


def print_evaluation(text, lm):
    """
    Runs the language model on the given text and prints three metrics: log probability of the text under this model
    (treating the text as one log sequence), average log probability (the previous value divided by sequence length),
    and perplexity (averaged "branching favor" of the model)
    :param text: the text to evaluate
    :param lm: model to evaluate
    """
    log_prob = lm.get_log_prob_sequence(text, " ")
    print("Log prob of text %f" % log_prob)
    print("Avg log prob: %f" % (log_prob/len(text)))
    perplexity = np.exp(-log_prob/len(text))
    print("Perplexity: %f" % perplexity)


if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)

    train_text = read_text(args.train_path)
    dev_text = read_text(args.dev_path)

    # Vocabs is lowercase letters a to z and space
    vocab = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
    vocab_index = Indexer()
    for char in vocab:
        vocab_index.add_and_get_index(char)
    print(repr(vocab_index))

    print("First 100 characters of train:")
    print(train_text[0:100])
    system_to_run = args.model
    # Train our model
    if system_to_run == "RNN":
        model = train_lm(args, train_text, dev_text, vocab_index)
    elif system_to_run == "UNIFORM":
        model = UniformLanguageModel(len(vocab))
    else:
        raise Exception("Pass in either UNIFORM or LSTM to run the appropriate system")

    print_evaluation(dev_text, model)

