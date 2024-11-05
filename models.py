# models.py

import numpy as np
import collections
import torch
import torch.nn as nn
import random
from matplotlib import pyplot as plt
from torch.ao.nn.quantized.functional import leaky_relu

plt.style.use('ggplot')
import os
from utils import Indexer
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


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
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
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
    def __init__(self, input_size, unique_charactor_amount, hidden_size, hidden_layer1, hidden_layer2, vocab_index, device='cpu'):
        super(RNNClassifier, self).__init__()
        self.charactor_embedding = nn.Embedding(num_embeddings=unique_charactor_amount, embedding_dim=input_size)
        self.lstm = nn.LSTM(input_size=input_size,  # embedding dimension
                            hidden_size=hidden_size,  # Number of LSTM units
                            num_layers=1,  # Number of LSTM layers
                            batch_first=True,
                            )  # Input shape will be [batch_size, seq_length, input_size]
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, hidden_layer1)
        self.linear2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.linear3 = nn.Linear(hidden_layer2, 2)
        self.vocab_index = vocab_index
        self.device = device

        self.to(device)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        embedded_vector = self.charactor_embedding(x)
        output, (hidden, cell) = self.lstm(embedded_vector)

        hidden = hidden.squeeze()

        hidden = hidden.squeeze(0)

        val = self.linear(hidden)

        val = self.relu(val)

        val = self.dropout(val)

        val = self.linear2(val)

        val = self.relu(val)

        val = self.dropout(val)

        predicted_val = self.linear3(val)

        return predicted_val

    def predict(self, context):
        index_string = torch.tensor([self.vocab_index.index_of(x) for x in context], dtype=torch.long,
                                    device=self.device)
        predicted = self.forward(index_string)
        predicted_class = torch.argmax(predicted, dim=-1)  # Add dimension for argmax
        return predicted_class


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def raw_string_to_indices(train_cons, train_vowel, vocab_index):
    cons_data = [[train_cons[index], 0, index] for index in range(0, len(train_cons))]
    vowels_data = [[train_vowel[index], 1, len(train_cons) + index] for index in range(0, len(train_vowel))]

    all_data = cons_data + vowels_data

    for index in all_data:
        index_string = [vocab_index.index_of(x) for x in index[0]]
        index.append(index_string)

    return all_data


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    try:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = raw_string_to_indices(train_cons_exs, train_vowel_exs, vocab_index)
        test_data = raw_string_to_indices(dev_cons_exs, dev_vowel_exs, vocab_index)

        n_samples = len(data)
        n_test_samples = len(dev_cons_exs) + len(dev_vowel_exs)
        epochs = 15
        batch_size = 4
        input_dim_size = 20
        hidden_size = 40
        hidden_layer1 = 16
        hidden_layer2 = 8
        learning_rate = 0.0005

        train_los_ar = []
        train_accuracy_ar = []
        dev_loss_ar = []
        dev_accuracy_ar = []
        epoch_array = np.array([x for x in range(1, epochs + 1)])

        unique_charactor_amount = vocab_index.__len__()

        rnn_classification_model = RNNClassifier(input_size=input_dim_size, unique_charactor_amount=unique_charactor_amount,
                                                 hidden_size=hidden_size, hidden_layer1=hidden_layer1,
                                                 hidden_layer2=hidden_layer2, vocab_index=vocab_index, device=str(device))

        loss_function = nn.CrossEntropyLoss().to(device)

        optimizer = torch.optim.Adam(rnn_classification_model.parameters(), lr=learning_rate, weight_decay=0.0001)

        for epoch in range(epochs):
            rnn_classification_model.train()
            total_loss = 0
            train_correct = 0
            random.shuffle(data)

            for i in range(0, n_samples, batch_size):
                batch_data = data[i:min(i + batch_size, n_samples)]

                batch_index_data = torch.LongTensor([x[3] for x in batch_data]).to(device)
                batch_label = torch.LongTensor([x[1] for x in batch_data]).to(device)

                optimizer.zero_grad()
                y = rnn_classification_model(batch_index_data)

                # Ensure outputs and labels have correct shape
                if batch_size == 1:
                    # Add batch dimension
                    y = y.unsqueeze(0)

                loss = loss_function(y, batch_label)
                total_loss += loss.item()

                _, train_predicted = torch.max(y.data, 1)

                train_correct += (train_predicted == batch_label).sum().item()

                loss.backward()
                optimizer.step()

            # Calculate average loss properly accounting for possibly incomplete final batch
            n_batches = (n_samples + batch_size - 1) // batch_size
            avg_loss = total_loss / n_batches
            train_accuracy = train_correct / n_samples

            train_los_ar.append(avg_loss)
            train_accuracy_ar.append(train_accuracy)

            # Evaluation phase
            rnn_classification_model.eval()
            total_test_loss = 0
            test_correct = 0

            with torch.no_grad():
                for i in range(0, n_test_samples, batch_size):
                    batch_test_data = test_data[i:min(i + batch_size, n_test_samples)]

                    batch_test_index_data = torch.LongTensor([x[3] for x in batch_test_data]).to(device)
                    batch_test_label = torch.LongTensor([x[1] for x in batch_test_data]).to(device)

                    y = rnn_classification_model(batch_test_index_data)

                    if batch_size == 1:
                        # Add batch dimension
                        y = y.unsqueeze(0)

                    test_loss = loss_function(y, batch_test_label)

                    total_test_loss += test_loss.item()

                    _, test_predicted = torch.max(y.data, 1)
                    test_correct += (test_predicted == batch_test_label).sum().item()

                n_test_batches = (n_test_samples + batch_size - 1) // batch_size
                avg_test_loss = total_test_loss / n_test_batches
                test_accuracy = test_correct / n_test_samples

                dev_loss_ar.append(avg_test_loss)
                dev_accuracy_ar.append(test_accuracy)

            print(f'Epoch {epoch + 1}:')
            print(f'Training Loss: {avg_loss:.4f}')
            print(f'Test Loss: {avg_test_loss:.4f}')
            print(f'Training Accuracy: {train_accuracy:.3f}')
            print(f'Test Accuracy: {test_accuracy :.3f}')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.tight_layout(pad=3.0)

        # Plot Loss curves in the first subplot
        ax1.plot(epoch_array, train_los_ar, label='Train Loss')
        ax1.plot(epoch_array, dev_loss_ar, label='Dev Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title('Training and Development Loss')

        # Plot Accuracy curves in the second subplot
        ax2.plot(epoch_array, train_accuracy_ar, label='Train Accuracy')
        ax2.plot(epoch_array, dev_accuracy_ar, label='Dev Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        ax2.set_title('Training and Development Accuracy')

        plt.savefig('classifier_accuracy_loss.png')

        #Saving the trained model
        os.makedirs('trained_models', exist_ok=True)
        model_save_path = os.path.join('trained_models', 'rnn_binary_classifier.pth')
        torch.save(rnn_classification_model, model_save_path)
        print(f'Model saved to {model_save_path}')

        return rnn_classification_model

    except Exception as e:
        print(f"Unexpected error in training process: {str(e)}")
        raise


#####################
# MODELS FOR PART 2 #
#####################


class LanguageModel(object):

    def get_log_prob_single(self, next_char, context):
        """
        Scores one character following the given context. That is, returns
        log P(next_char | context)
        The log should be base e
        :param next_char:
        :param context: a single character to score
        :return:
        """
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context):
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return:
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_log_prob_single(self, next_char, context):
        return np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel, nn.Module):
    def __init__(self, model_emb, model_dec, vocab_index, num_layers=1, dropout_value=0.5, device='cpu'):
        super(RNNLanguageModel, self).__init__()
        self.model_emb = model_emb  # embedding dimension
        self.model_dec = model_dec  # hidden layer size
        self.vocab_index = vocab_index  # vocab index
        self.device = device

        self.num_layers = num_layers

        self.vocab_size = vocab_index.__len__()

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size + 1, embedding_dim=model_emb)

        self.lstm = nn.LSTM(input_size=model_emb,
                            hidden_size=model_dec,
                            num_layers=num_layers,
                            dropout=dropout_value if num_layers > 1 else 0,
                            batch_first=True)

        self.linear = nn.Linear(model_dec, self.vocab_size - 1)

        self.dropout = nn.Dropout(dropout_value)

        self.softmax = nn.LogSoftmax(dim=-1)

        self.to(device)

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.num_layers, batch_size, self.model_dec, device=self.device),
                torch.zeros(self.num_layers, batch_size, self.model_dec, device=self.device))

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0))

        embedded = self.embedding(x)

        output, (hidden, cell) = self.lstm(embedded, hidden)

        output = self.dropout(output)

        linear = self.linear(output)

        final_output = self.softmax(linear)

        return final_output, (hidden, cell)

    def get_log_prob_single(self, next_char, context, hidden=None):
        self.eval()

        # Add SOS token at the start if context is empty
        if not context:
            context_indices = [self.vocab_index.index_of("sos")]
        else:
            context_indices = [self.vocab_index.index_of(x) for x in context]

        context_text = torch.tensor([context_indices], dtype=torch.long).to(self.device)

        print(context_text.shape)

        with torch.no_grad():
            y_output, hidden_output = self.forward(context_text, hidden)

            last_charactor_predicted = y_output[0, -1]

            print(last_charactor_predicted.shape)

            print(last_charactor_predicted)

            next_charactor_actual_index = self.vocab_index.index_of(next_char)

            log_probability = last_charactor_predicted[next_charactor_actual_index]

            print("predicted : ", self.vocab_index.get_object(torch.argmax(last_charactor_predicted).cpu().item()),
                  "actual", next_char)

            return log_probability, hidden_output

    def get_log_prob_sequence(self, next_chars, context):
        total_log_probability = 0

        current_context = context

        hidden = None

        for index in range(0, len(next_chars)):
            log_probability, hidden = self.get_log_prob_single(next_chars[index], current_context, hidden)

            total_log_probability += log_probability

            current_context = current_context[1:] + next_chars[index]

        return total_log_probability.cpu().item()


def chunk_required_data(text, chunk_size, vocab_index, overlap_size=1):
    chunks_extracted = []
    target_extracted = []

    for index in range(0, len(text) - chunk_size, overlap_size):
        chunks_extracted.append(text[index:index + chunk_size - 1])
        target_extracted.append(text[index:index + chunk_size])

    chunks_extracted, target_extracted = extract_required_indices(chunks_extracted, target_extracted, vocab_index)

    return chunks_extracted, target_extracted


def extract_required_indices(extracted_text, extracted_target, vocab_index):
    # Prepend SOS token to each sequence instead of the whole list
    text_indices = np.asarray([[vocab_index.index_of("sos")] + [vocab_index.index_of(x) for x in seq]
                               for seq in extracted_text])
    target_indices = np.asarray([[vocab_index.index_of(x) for x in index]
                                 for index in extracted_target])

    return text_indices, target_indices


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """

    chunk_size = 20
    overlap_size = 5
    learning_rate = 0.002
    epochs = 10
    batch_size = 8
    burn_in_length = 5
    lstm_layer_count = 1
    embedding_size = 16
    hidden_layers = 40

    vocab_index.add_and_get_index("sos")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chunked_train_text, target_train = chunk_required_data(train_text, chunk_size, vocab_index, overlap_size)
    chunked_dev_text, target_test = chunk_required_data(dev_text, chunk_size, vocab_index, overlap_size)

    train_los_ar = []
    dev_loss_ar = []
    epoch_array = np.array([x for x in range(1, epochs + 1)])

    train_chunks = torch.from_numpy(chunked_train_text).long().to(device)
    train_targets = torch.from_numpy(target_train).long().to(device)
    dev_chunks = torch.from_numpy(chunked_dev_text).long().to(device)
    dev_targets = torch.from_numpy(target_test).long().to(device)

    language_model = RNNLanguageModel(model_emb=embedding_size, model_dec=hidden_layers, vocab_index=vocab_index,
                                      num_layers=lstm_layer_count, device=str(device))
    loss_function = nn.NLLLoss().to(device)  # Negative log likelihood
    optimizer = torch.optim.Adam(language_model.parameters(), lr=learning_rate)

    # for index in range(0, len(chunked_train_text)):
    #     print(chunked_train_text[index],  target_train[index])
    #
    # # for index in range(0, len(chunked_dev_text)):
    # #     print(chunked_dev_text[index],  target_test[index])

    print("Train text: ", len(chunked_train_text))
    print("Dev text: ", len(chunked_dev_text))
    print("Vocab size: ", vocab_index.__len__())

    for epoch in range(epochs):
        total_loss = 0
        language_model.train()

        for i in range(0, len(train_chunks), batch_size):
            batch_chunks = train_chunks[i:i + batch_size]
            batch_targets = train_targets[i:i + batch_size]

            optimizer.zero_grad()

            if burn_in_length > 0:
                with torch.no_grad():
                    # Only use burn_in_length characters for burn-in
                    burn_in_input = batch_chunks[:, :burn_in_length]
                    burn_out, hidden = language_model(burn_in_input)

                    # print("Burn in input : ", burn_in_input, "Burn out : ", burn_out)

                model_input = batch_chunks[:, burn_in_length:]
                batch_targets = batch_targets[:, burn_in_length:]
            else:
                model_input = batch_chunks

            # Detach hidden state for next batch
            hidden = (hidden[0].detach(), hidden[1].detach())

            y_output, hidden = language_model(model_input, hidden)

            y_output = y_output.view(-1, vocab_index.__len__() - 1)
            batch_targets = batch_targets.reshape(-1)

            loss = loss_function(y_output, batch_targets)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            # print(f"Epoch : {epoch + 1}", [torch.argmax(x).cpu().item() for x in y_output], batch_targets)

        language_model.eval()
        with torch.no_grad():
            dev_total_loss = 0
            for i in range(0, len(dev_chunks), batch_size):
                batch_dev_chunks = dev_chunks[i:i + batch_size]
                batch_dev_targets = dev_targets[i:i + batch_size]

                if burn_in_length > 0:
                    # Only use burn_in_length characters for burn-in
                    burn_in_input = batch_dev_chunks[:, :burn_in_length]
                    burn_out, hidden = language_model(burn_in_input)

                    model_input = batch_dev_chunks[:, burn_in_length:]
                    batch_dev_targets = batch_dev_targets[:, burn_in_length:]
                else:
                    model_input = batch_dev_chunks
                    hidden = None

                batch_dev_targets = batch_dev_targets.reshape(-1)
                y_output, hidden = language_model(model_input, hidden)

                y_output = y_output.view(-1, vocab_index.__len__() - 1)

                dev_loss = loss_function(y_output, batch_dev_targets)

                dev_total_loss += dev_loss.item()

            n_train_batches = (len(train_chunks) + batch_size - 1) // batch_size
            n_test_batches = (len(dev_chunks) + batch_size - 1) // batch_size

            train_avg_loss = total_loss / n_train_batches
            train_los_ar.append(train_avg_loss)

            dev_avg_loss = dev_total_loss / n_test_batches
            dev_loss_ar.append(dev_avg_loss)

            print(f"Epoch : {epoch + 1}")
            print(f"Train Average Loss : {train_avg_loss}")
            print(f"Test Average Loss : {dev_avg_loss}")

    plt.plot(epoch_array, train_los_ar, label='Train Loss')
    plt.plot(epoch_array, dev_loss_ar, label='Dev Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Development Loss of the Language model')
    plt.legend()
    plt.savefig('language_model_loss.png')

    return language_model


#################
# VISUALIZATION #
#################

def run_experiment(max_context_length=20):
    DEV_CONS_PATH = r"data/dev-consonant-examples.txt"
    DEV_VOWEL_PATH = r"data/dev-vowel-examples.txt"
    MODEL_PATH = r"trained_models/rnn_binary_classifier.pth"

    if max_context_length not in range(1,21):
        print(f"Max context length should be in range [1, 20]")
        return

    def load_examples(filename):
        examples = []
        with open(filename, 'r') as f:
            for line in f:
                examples.append(line)
        return examples

    def load_rnn_classifier(model_path):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = torch.load(model_path, map_location=device)
            model.eval()
            print("Model loaded successfully.")
            return model
        except FileNotFoundError:
            print(f"Error: The model file '{model_path}' was not found.")
            raise

    def evaluate_model_on_context_lengths(model, consonant_examples, vowel_examples):
        accuracy_by_length = {}
        vowels = 'aeiou'

        for length in range(1, max_context_length + 1):
            if length == 20:
                trimmed_consonants = [ex[:length ] for ex in consonant_examples]
                trimmed_vowels = [ex[:length ] for ex in vowel_examples]
            else:
                trimmed_text = [ex[:length + 1] for ex in consonant_examples + vowel_examples]
                trimmed_vowels = []
                trimmed_consonants = []

                for ex in trimmed_text:
                    if ex[-1] in vowels:
                        trimmed_vowels.append(ex[:-1])
                    else:
                        trimmed_consonants.append(ex[:-1])

            # model evaluation
            correct_predictions = sum(1 for ex in trimmed_consonants if model.predict(ex) == 0)
            correct_predictions += sum(1 for ex in trimmed_vowels if model.predict(ex) == 1)
            total = len(trimmed_consonants) + len(trimmed_vowels)
            accuracy = correct_predictions / total

            accuracy_by_length[length] = accuracy
            print(f"Context length {length}: Accuracy = {accuracy:.3f}")

        return accuracy_by_length

    def visualize_accuracy_trend(accuracy_data, title):
        lengths = list(accuracy_data.keys())
        accuracies = list(accuracy_data.values())

        plt.figure(figsize=(10, 6))
        plt.plot(lengths, accuracies, marker='o')
        plt.title(title)
        plt.xlabel("Context Length")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.show()

    # Vocabulary and indexer
    vocab = [chr(ord('a') + i) for i in range(26)] + [' ']
    vocab_index = Indexer()
    for char in vocab:
        vocab_index.add_and_get_index(char)

    # Load testing data
    dev_cons_exs = load_examples(DEV_CONS_PATH)
    dev_vowel_exs = load_examples(DEV_VOWEL_PATH)

    # Load the trained model
    try:
        rnn_model = load_rnn_classifier(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: The model file '{MODEL_PATH}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # Evaluate and visualize
    print("\nEvaluating RNN Classifier")
    rnn_accuracy_data = evaluate_model_on_context_lengths(rnn_model, dev_cons_exs, dev_vowel_exs)
    visualize_accuracy_trend(rnn_accuracy_data, "RNN Classifier Accuracy vs. Context Length")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the consonant vowel classification experiment for different "
                                                 "context length.")
    parser.add_argument("--max_context_length", type=int, default=20, help="Maximum context length for evaluation.")

    # Parse arguments
    args = parser.parse_args()

    # Run the experiment with parsed arguments
    run_experiment(args.max_context_length)


if __name__ == "__main__":
    main()


