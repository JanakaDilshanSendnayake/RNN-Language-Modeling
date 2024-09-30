# lm_classifier.py

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
    parser.add_argument('--model', type=str, default='FREQUENCY', help='model to run (FREQUENCY or RNN)')
    parser.add_argument('--train_cons', type=str, default='data/train-consonant-examples.txt', help='path to train consonant examples')
    parser.add_argument('--train_vowel', type=str, default='data/train-vowel-examples.txt', help='path to train vowel examples')
    parser.add_argument('--dev_cons', type=str, default='data/dev-consonant-examples.txt', help='path to dev consonant examples')
    parser.add_argument('--dev_vowel', type=str, default='data/dev-vowel-examples.txt', help='path to dev vowel examples')
    args = parser.parse_args()
    return args


def read_examples(file):
    """
    :param file:
    :return: The text in the given file as a single string
    """
    all_lines = []
    for line in open(file):
        # Drop the last token (newline) but don't call strip() to keep whitespace
        all_lines.append(line[:-1])
        print(line[:-1])
    print("%i lines read in" % len(all_lines))
    return all_lines


def print_evaluation(dev_consonant_exs, dev_vowel_exs, model):
    """
    Runs the classifier on the given text
    :param text:
    :param lm:
    :return:
    """
    num_correct = 0
    for ex in dev_consonant_exs:
        if model.predict(ex) == 0:
            num_correct += 1
    for ex in dev_vowel_exs:
        if model.predict(ex) == 1:
            num_correct += 1
    num_total = len(dev_consonant_exs) + len(dev_vowel_exs)
    print("%i correct / %i total = %.3f percent accuracy" % (num_correct, num_total, float(num_correct)/num_total * 100.0))

if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)

    train_cons_exs = read_examples(args.train_cons)
    train_vowel_exs = read_examples(args.train_vowel)
    dev_cons_exs = read_examples(args.dev_cons)
    dev_vowel_exs = read_examples(args.dev_vowel)

    # Vocabs is lowercase letters a to z and space
    vocab = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
    vocab_index = Indexer()
    for char in vocab:
        vocab_index.add_and_get_index(char)
    print(repr(vocab_index))

    system_to_run = args.model
    # Train our model
    if system_to_run == "RNN":
        model = train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index)
    elif system_to_run == "FREQUENCY":
        model = train_frequency_based_classifier(train_cons_exs, train_vowel_exs)
    else:
        raise Exception("Pass in either UNIFORM or LSTM to run the appropriate system")

    print_evaluation(dev_cons_exs, dev_vowel_exs, model)
    # print_evaluation(train_cons_exs[0:50], train_vowel_exs[0:50], model)
