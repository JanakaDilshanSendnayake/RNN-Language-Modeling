# models.py

import numpy as np
import collections
import torch
import torch.nn as nn
import random
from matplotlib import pyplot as plt
import seaborn as sns
import os
from utils import Indexer
import argparse
import warnings
from sklearn.metrics import confusion_matrix
plt.style.use('ggplot')
warnings.filterwarnings("ignore", category=FutureWarning)

os.makedirs('trained_models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

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
    def __init__(self, input_size, unique_charactor_amount, hidden_size, hidden_layer1, hidden_layer2, vocab_index,
                 dropout, device='cpu'):
        super(RNNClassifier, self).__init__()

        # Embedding layer to convert character indices to dense vectors
        self.charactor_embedding = nn.Embedding(num_embeddings=unique_charactor_amount, embedding_dim=input_size)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size,  # embedding dimension
                            hidden_size=hidden_size,  # Number of LSTM units
                            num_layers=1,  # Number of LSTM layers
                            batch_first=True,
                            )  # Input shape will be [batch_size, seq_length, input_size]

        # Dropout layer
        self.dropout = nn.Dropout(p= dropout)

        # Relu activation function
        self.relu = nn.ReLU()

        # First dense layer
        self.linear = nn.Linear(hidden_size, hidden_layer1)

        # Second dense layer
        self.linear2 = nn.Linear(hidden_layer1, hidden_layer2)

        # Third dense layer
        self.linear3 = nn.Linear(hidden_layer2, 2)
        self.vocab_index = vocab_index
        self.device = device

        # Assign the device to train the model (GPU or CPU)
        self.to(device)

    def forward(self, x):
        """Forward function for the RNN model."""
        # If the input is a single batch, add a batch dimension
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # Convert the converted indices to the vector embeddings
        embedded_vector = self.charactor_embedding(x)
        output, (hidden, cell) = self.lstm(embedded_vector)

        # Remove if there is any extra dimension
        hidden = hidden.squeeze()

        # Apply the fully connected layer

        # Apply the dense layer
        val = self.linear(hidden) # First dense layer
        val = self.relu(val) # Relu activation function
        val = self.dropout(val) # Dropout layer


        val = self.linear2(val) # Second dense layer
        val = self.relu(val) # Relu activation function
        val = self.dropout(val) # Dropout layer

        predicted_val = self.linear3(val) #  Final dense layer

        # Didn't apply the softmax activation at the end because,
        # in pytorch it will be added in the loss function default if the loss function is categorical cross entropy

        return predicted_val

    def predict(self, context):
        """This function predicts the class of the given context (a single string)."""

        # Convert the context to the indices (0 to 26)
        index_string = torch.tensor([self.vocab_index.index_of(x) for x in context], dtype=torch.long,
                                    device=self.device)

        # get the models prediction using the forward pass
        predicted = self.forward(index_string)

        # Get the class with the highest probability using argmax since the final activation is softmax
        predicted_class = torch.argmax(predicted, dim=-1)  # Add dimension for argmax

        # return the predicted class
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
    """Convert the raw string data to indices using the vocab index. and return it as a list"""

    # Create a list of the consonant strings with the class label 0  and the index of the consonant string
    cons_data = [[train_cons[index], 0, index] for index in range(0, len(train_cons))]

    # Create a list of the vowel strings with the class label 1  and the index of the vowel string
    vowels_data = [[train_vowel[index], 1, len(train_cons) + index] for index in range(0, len(train_vowel))]

    # Combine the consonant and vowel data
    all_data = cons_data + vowels_data

    # Convert the string data to the indices using the vocab index
    for index in all_data:
        index_string = [vocab_index.index_of(x) for x in index[0]]
        index.append(index_string)

    # return the lists that include, string, class label, index and the converted indices
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

        # set the device, it  will be 'cuda' if a GPU is available, otherwise 'cpu'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert the raw string data to indices using the vocab index for the training and testing data
        data = raw_string_to_indices(train_cons_exs, train_vowel_exs, vocab_index)
        test_data = raw_string_to_indices(dev_cons_exs, dev_vowel_exs, vocab_index)

        # Set the  required hyperparameters
        n_samples = len(data)         # Number of samples in the training data
        n_test_samples = len(dev_cons_exs) + len(dev_vowel_exs) # Number of samples in the testing data
        epochs = 15            # Assign the epoch amount
        batch_size = 4         # Assign the batch size
        input_dim_size = 20    # Assign the embedding dimension size
        hidden_size = 40       # Assign the hidden state size
        hidden_layer1 = 32     # Assign the hidden layer1 size
        hidden_layer2 = 8      # Assign the hidden layer2 size
        learning_rate = 0.0005  # Assign the learning rate
        dropout_rate = 0.2      # Assign the dropout rate

        # Lists to track both training and dev loss and accuracy
        train_los_ar = []
        train_accuracy_ar = []
        dev_loss_ar = []
        dev_accuracy_ar = []
        epoch_array = np.array([x for x in range(1, epochs + 1)])

        # Get the unique charactor amount from the vocab index (get the length of the charactor dictionary)
        unique_charactor_amount = vocab_index.__len__()


        # Initialize the RNN model
        rnn_classification_model = RNNClassifier(input_size=input_dim_size, unique_charactor_amount=unique_charactor_amount,
                                                 hidden_size=hidden_size, hidden_layer1=hidden_layer1,
                                                 hidden_layer2=hidden_layer2, vocab_index=vocab_index,dropout=dropout_rate,
                                                 device=str(device))

        # Assign loss function as CrossEntropyLoss
        loss_function = nn.CrossEntropyLoss().to(device)

        # Assign the optimizer as Adam optimizer with learning rate and a weight decay (to handle overfitting)
        optimizer = torch.optim.Adam(rnn_classification_model.parameters(), lr=learning_rate, weight_decay=0.0001)

        true_labels = []
        predicted_labels = []

        # Start the training process
        for epoch in range(epochs):
            rnn_classification_model.train() # Set the model to training mode
            total_loss = 0     # Assign total loss 0 for each starting epoch
            train_correct = 0  # Assign total correct predictions 0 for each starting epoch
            random.shuffle(data) # Shuffle the data for each epoch since the data is ordered

            #  Batch training for the epoch
            for i in range(0, n_samples, batch_size):
                batch_data = data[i:min(i + batch_size, n_samples)]

                # Get the indices and the labels of the batch data
                batch_index_data = torch.LongTensor([x[3] for x in batch_data]).to(device)
                batch_label = torch.LongTensor([x[1] for x in batch_data]).to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Get the model prediction
                y = rnn_classification_model(batch_index_data)

                # If the batch size is 1, add a batch dimension
                if batch_size == 1:
                    # Add batch dimension
                    y = y.unsqueeze(0)

                # Calculate the loss
                loss = loss_function(y, batch_label)

                # Add the loss to the total loss
                total_loss += loss.item()

                # Get the predicted class
                _, train_predicted = torch.max(y.data, 1)

                # Calculate the correct predictions
                train_correct += (train_predicted == batch_label).sum().item()

                # Execute the back propagation
                loss.backward()

                # Update the weights
                optimizer.step()

            # Calculate average loss
            n_batches = (n_samples + batch_size - 1) // batch_size # Calculate the number of batches
            avg_loss = total_loss / n_batches                  # Calculate the average loss
            train_accuracy = train_correct / n_samples        # Calculate the training accuracy

            train_los_ar.append(avg_loss)  # Append the average loss to the list
            train_accuracy_ar.append(train_accuracy)   # Append the training accuracy to the list

            # Evaluation phase
            rnn_classification_model.eval()  # Set the model to evaluation mode
            total_test_loss = 0   # Assign total loss 0
            test_correct = 0      # Assign total correct predictions 0

            # Batch testing for the epoch
            with torch.no_grad():           # blocked calculating gradients
                for i in range(0, n_test_samples, batch_size):   # Loop through the testing data
                    batch_test_data = test_data[i:min(i + batch_size, n_test_samples)]

                    # Prepare test batch data
                    batch_test_index_data = torch.LongTensor([x[3] for x in batch_test_data]).to(device)
                    batch_test_label = torch.LongTensor([x[1] for x in batch_test_data]).to(device)

                    # Get the model prediction
                    y = rnn_classification_model(batch_test_index_data)

                    # If the batch size is 1, add a batch dimension
                    if batch_size == 1:
                        # Add batch dimension
                        y = y.unsqueeze(0)

                    # Calculate the test loss
                    test_loss = loss_function(y, batch_test_label)

                    # Add the loss to the total loss
                    total_test_loss += test_loss.item()

                    # Get the predicted class
                    _, test_predicted = torch.max(y.data, 1)
                    test_correct += (test_predicted == batch_test_label).sum().item()

                    if epoch == epochs - 1:
                        true_labels.extend(batch_test_label.cpu().numpy())
                        predicted_labels.extend(test_predicted.cpu().numpy())


                n_test_batches = (n_test_samples + batch_size - 1) // batch_size # Calculate the number of batches
                avg_test_loss = total_test_loss / n_test_batches     # Calculate the average loss
                test_accuracy = test_correct / n_test_samples        # Calculate the test accuracy

                dev_loss_ar.append(avg_test_loss)  # Append the average loss to the list
                dev_accuracy_ar.append(test_accuracy) # Append the test accuracy to the list

            print(f'Epoch {epoch + 1}:') # Print the epoch number
            print(f'Training Loss: {avg_loss:.4f}')    # Print the training loss
            print(f'Test Loss: {avg_test_loss:.4f}')   # Print the test loss
            print(f'Training Accuracy: {train_accuracy:.3f}')  # Print the training accuracy
            print(f'Test Accuracy: {test_accuracy :.3f}')    # Print the test accuracy

        # Compute confusion matrix
        class_names = ["Consonant", "Vowel"]
        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"Confusion Matrix")
        save_path = os.path.join('plots', f'confusion_matrix.png')
        plt.savefig(save_path)
        plt.close()

        # Plot the accuracy and loss curves
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
        save_path = os.path.join('plots', 'classifier_accuracy_loss.png')
        # Saved the plot figure
        plt.savefig(save_path)

        # Saving the trained model
        model_save_path = os.path.join('trained_models', 'rnn_binary_classifier.pth')
        torch.save(rnn_classification_model, model_save_path)
        print(f'Model saved to {model_save_path}')
        run_experiment(max_context_length=20, print_ok=False)
        # Return the trained model
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
        self.device = device       # set the device cpu or gpu to train the model

        self.num_layers = num_layers  # set number of hidden layers for lstm

        self.vocab_size = vocab_index.__len__()  # Get the dictionary size of the index (27 due to start of sequence tag is added)

        # Embedding layer to convert character indices to dense vectors
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size + 1, embedding_dim=model_emb)

        # LSTM layer, dropout is applied if number of layers are greater than 1
        self.lstm = nn.LSTM(input_size=model_emb,
                            hidden_size=model_dec,
                            num_layers=num_layers,
                            dropout=dropout_value if num_layers > 1 else 0,
                            batch_first=True)

        # Linear layer to convert the hidden layer to the vocab size
        self.linear = nn.Linear(model_dec, self.vocab_size - 1)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_value)

        # Log softmax is used as the activation for the final layer since it
        self.softmax = nn.LogSoftmax(dim=-1)

        # Assign the device to train the model (GPU or CPU)
        self.to(device)

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.num_layers, batch_size, self.model_dec, device=self.device),
                torch.zeros(self.num_layers, batch_size, self.model_dec, device=self.device))

    def forward(self, x, hidden=None):
        """ Processes the input tensor through embedding, LSTM layers, dropout,
        and a linear layer to produce log probabilities"""

        # If the hidden state is None, initialize it
        if hidden is None:
            hidden = self.init_hidden(x.size(0))

        # Convert the converted indices to the vector embeddings
        embedded = self.embedding(x)

        # Send the embedding through LSTM
        output, (hidden, cell) = self.lstm(embedded, hidden)

        # Apply the dropout for output
        output = self.dropout(output)

        # Linear layer to get the vector of the vocabulary size
        linear = self.linear(output)

        # convert it to log prob probabilities by applying log softmax activation
        final_output = self.softmax(linear)

        # return final output layer and the hidden states
        return final_output, (hidden, cell)

    def get_log_prob_single(self, next_char, context, hidden=None):
        """Calculate the log probability of a single character given the context"""

        # Set the model to evaluation mode
        self.eval()

        # Add SOS token at the start if context is empty
        if not context:
            context_indices = [self.vocab_index.index_of("sos")]
        else:
            context_indices = [self.vocab_index.index_of(x) for x in context]

        # convert the context text to a tensor and move it to device
        context_text = torch.tensor([context_indices], dtype=torch.long).to(self.device)

        with torch.no_grad():
            # Get the output and hidden states
            y_output, hidden_output = self.forward(context_text, hidden)

            # Get the last character predicted
            last_charactor_predicted = y_output[0, -1]

            print(last_charactor_predicted.shape)

            print(last_charactor_predicted)

            # Get the index of the next character
            next_charactor_actual_index = self.vocab_index.index_of(next_char)

            # Get the log probability of the last character predicted
            log_probability = last_charactor_predicted[next_charactor_actual_index]

            print("predicted : ", self.vocab_index.get_object(torch.argmax(last_charactor_predicted).cpu().item()),
                  "actual", next_char)

            # return the log probability and the hidden states
            return log_probability, hidden_output

    def get_log_prob_sequence(self, next_chars, context):
        """Calculate the log probability of a character given the context"""

        # Assign log probability as 0
        total_log_probability = 0

        # Assign the current context
        current_context = context

        # Assign the hidden state as None
        hidden = None

        # process each character in the next characters
        for index in range(0, len(next_chars)):
            # Get the log probability of the next character
            log_probability, hidden = self.get_log_prob_single(next_chars[index], current_context, hidden)

            # Add the log probability to the total log probability
            total_log_probability += log_probability

            # Update the context by removing the first charactor in previous and adding the predicted charactor
            current_context = current_context[1:] + next_chars[index]

        return total_log_probability.cpu().item()


def chunk_required_data(text, chunk_size, vocab_index, overlap_size=1):
    """Split the text into overlapping chunks and extract their indices"""

    chunks_extracted = []
    target_extracted = []

    # The process of creating overlapping chunks
    for index in range(0, len(text) - chunk_size, overlap_size):
        chunks_extracted.append(text[index:index + chunk_size - 1])
        target_extracted.append(text[index:index + chunk_size])

    # Get the extracted indices of the selected chunks and the target value of each chunk
    chunks_extracted, target_extracted = extract_required_indices(chunks_extracted, target_extracted, vocab_index)

    # Return both chunk extracted and chunk values
    return chunks_extracted, target_extracted


def extract_required_indices(extracted_text, extracted_target, vocab_index):
    """Convert text sequences and target sequences into index representation 
    using the vocabulary index"""

    # Add SOS (Start of Sequence) token to each sequence instead of the whole list
    text_indices = np.asarray([[vocab_index.index_of("sos")] + [vocab_index.index_of(x) for x in seq]
                               for seq in extracted_text])
    # Get the target value of the indices
    target_indices = np.asarray([[vocab_index.index_of(x) for x in index]
                                 for index in extracted_target])

    # Return the indices of the text and the target values
    return text_indices, target_indices


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    try:
        chunk_size = 20   # Assign the text chunking size
        overlap_size = 2   # Assign the text overlapping size
        learning_rate = 0.001  # Assign the learning rate
        epochs = 15  # Assign the epoch amount
        batch_size =  8    # Assign the batch size
        burn_in_length = 5  # Assign the burn in length
        lstm_layer_count = 2 # Assign the number of LSTM layers
        embedding_size = 16  # Assign the embedding size
        hidden_layers = 128  # Assign the hidden layer size

        # Add the start of sequence token to the vocabulary
        vocab_index.add_and_get_index("sos")

        # set the device as cuda if gpu is available, otherwise cpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # chunk and get the target indices of the train and dev text
        chunked_train_text, target_train = chunk_required_data(train_text, chunk_size, vocab_index, overlap_size)
        chunked_dev_text, target_test = chunk_required_data(dev_text, chunk_size, vocab_index, overlap_size)

        # Lists to store loss values for plotting
        train_los_ar = []  # Train loss list
        dev_loss_ar = []   # Dev loss list

        # Array to store the epoch values
        epoch_array = np.array([x for x in range(1, epochs + 1)])

        # Convert the numpy arrays to torch tensors and move them to the device
        train_chunks = torch.from_numpy(chunked_train_text).long().to(device) # train chunks tensor
        train_targets = torch.from_numpy(target_train).long().to(device) # train target tensor
        dev_chunks = torch.from_numpy(chunked_dev_text).long().to(device) # dev chunks tensor
        dev_targets = torch.from_numpy(target_test).long().to(device)  # dev target tensor

        # Initialize the RNN language model
        language_model = RNNLanguageModel(model_emb=embedding_size, model_dec=hidden_layers, vocab_index=vocab_index,
                                          num_layers=lstm_layer_count, device=str(device))

        # Assign the loss function as Negative log likelihood as the loss function
        loss_function = nn.NLLLoss().to(device)

        # Assign the optimizer as Adam optimizer with learning rate
        optimizer = torch.optim.Adam(language_model.parameters(), lr=learning_rate)

        print("Train text: ", len(chunked_train_text))
        print("Dev text: ", len(chunked_dev_text))
        print("Vocab size: ", vocab_index.__len__())

        hidden = None
        test_hidden = None
        # Evaluation process on the development set
        for epoch in range(epochs):
            total_loss = 0 # Assign total training loss 0 for each starting epoch
            language_model.train() # Set the model to training mode

            for i in range(0, len(train_chunks), batch_size):
                batch_chunks = train_chunks[i:i + batch_size]  # Get the batch chunks for training
                batch_targets = train_targets[i:i + batch_size] # Get the batch targets for training

                # Zero the gradients before the forward pass
                optimizer.zero_grad()

                # Handle burn-in for training set
                if burn_in_length > 0:
                    with torch.no_grad():
                        # Only use burn_in_length characters for burn-in
                        burn_in_input = batch_chunks[:, :burn_in_length]
                        burn_out, hidden = language_model(burn_in_input)

                    model_input = batch_chunks[:, burn_in_length:]
                    batch_targets = batch_targets[:, burn_in_length:]
                else:
                    model_input = batch_chunks

                if hidden is not None:
                    # Detach hidden state for next batch
                    hidden = (hidden[0].detach(), hidden[1].detach())

                    # Apply the forward pass to the model input  and get the predicted tensor
                    y_output, hidden = language_model(model_input, hidden)
                else:
                    # Apply the forward pass to the model input  and get the predicted tensor
                    y_output, hidden = language_model(model_input)

                # reshape the predicted tensor
                y_output = y_output.view(-1, vocab_index.__len__() - 1)

                # reshape the target tensor
                batch_targets = batch_targets.reshape(-1)

                # Calculate the loss
                loss = loss_function(y_output, batch_targets)

                # Add the loss to the total loss
                total_loss += loss.item()

                # Execute the back propagation
                loss.backward()

                # Update the weights
                optimizer.step()


            language_model.eval()
            with torch.no_grad():
                dev_total_loss = 0 # Assign total testing loss 0 for each starting epoch
                for i in range(0, len(dev_chunks), batch_size):
                    batch_dev_chunks = dev_chunks[i:i + batch_size] # Get the batch chunks for testing
                    batch_dev_targets = dev_targets[i:i + batch_size] # Get the batch targets for testing

                    # Handle burn-in for dev set
                    if burn_in_length > 0:
                        burn_in_input = batch_dev_chunks[:, :burn_in_length]  # burn in input
                        burn_out, test_hidden = language_model(burn_in_input)  # burn in output and hidden state

                        model_input = batch_dev_chunks[:, burn_in_length:]  # model input after burn in
                        batch_dev_targets = batch_dev_targets[:, burn_in_length:]  # model output target value
                    else:
                        model_input = batch_dev_chunks

                    # reshape dev targets to 1 dimension
                    batch_dev_targets = batch_dev_targets.reshape(-1)

                    if test_hidden is not None:
                        # Detach hidden state for next batch
                        test_hidden = (test_hidden[0].detach(), test_hidden[1].detach())

                        # Apply the forward pass to the model input  and get the predicted tensor
                        y_output, test_hidden = language_model(model_input, test_hidden)
                    else:
                        y_output, test_hidden = language_model(model_input)

                    # reshape the predicted tensor
                    y_output = y_output.view(-1, vocab_index.__len__() - 1)

                    # Calculate the test loss
                    dev_loss = loss_function(y_output, batch_dev_targets)

                    # Add the loss to the total loss
                    dev_total_loss += dev_loss.item()

                # Calculate the batch amount for both training and testing data
                n_train_batches = (len(train_chunks) + batch_size - 1) // batch_size
                n_test_batches = (len(dev_chunks) + batch_size - 1) // batch_size

                # Calculate the train average loss and store it
                train_avg_loss = total_loss / n_train_batches
                train_los_ar.append(train_avg_loss)

                # Calculate the test average loss and store it
                dev_avg_loss = dev_total_loss / n_test_batches
                dev_loss_ar.append(dev_avg_loss)

                # Print the average loss values of each epoch
                print(f"Epoch : {epoch + 1}")
                print(f"Train Average Loss : {train_avg_loss}")
                print(f"Test Average Loss : {dev_avg_loss}")

        # Plot training and testing loss
        plt.plot(epoch_array, train_los_ar, label='Train Loss')
        plt.plot(epoch_array, dev_loss_ar, label='Dev Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Development Loss of the Language model')
        plt.legend()
        save_path = os.path.join('plots', 'language_model_loss.png')
        plt.savefig(save_path)

        # Return the language model
        return language_model

    except Exception as e:
        print(f"Unexpected error in training process: {str(e)}")
        raise


#################
# VISUALIZATION #
#################

def run_experiment(max_context_length=20, print_ok=True):
    # Paths to the datasets and the trained model
    DEV_CONS_PATH = r"data/dev-consonant-examples.txt"
    DEV_VOWEL_PATH = r"data/dev-vowel-examples.txt"
    MODEL_PATH = r"trained_models/rnn_binary_classifier.pth"

    # Ensure the max_context_length is within valid range [1, 20]
    if max_context_length not in range(1, 21):
        print(f"Max context length should be in range [1, 20]")
        return

    # Function to load text
    def load_examples(filename):
        examples = []
        with open(filename, 'r') as f:
            for line in f:
                examples.append(line)
        return examples

    # Helper function to load the trained RNN model
    def load_rnn_classifier(model_path):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
            model = torch.load(model_path, map_location=device)  # Load the model
            model.eval()  # Set the model to evaluation mode
            print("Model loaded successfully.")
            return model
        except FileNotFoundError:
            print(f"Error: The model file '{model_path}' was not found.")
            raise

    def evaluate_model_on_context_lengths(model, consonant_examples, vowel_examples):
        """Evaluate the model on different context length, calculating accuracy and analyzing class imabalances.
        
        :param model: Model used to predict vowel or consonant classification
        :param consonant_examples: List of examples classified as consonants
        :param vowel_examples: List of examples classified as vowels

        :return accuracy_by_length: Dictionary with context lengths as keys and accuracy as values
        :return class_imbalances_data: Dictionary containing data for class imbalance visualization
        """

        accuracy_by_length = {}  # Dictionary to store accuracy for each context length
        vowels = 'aeiou'

        context_lengths = []   # List to store context lengths
        vowel_counts = []      # List to store the count of vowel examples
        consonant_counts = []  # List to store the count of consonant examples

        #  Loop through all context lengths from 1 to max context length
        for length in range(1, max_context_length + 1):
            # Trim the examples to the current context length
            if length == 20:
                # When the context length is 20, we don't need to classify the examples as they are
                # from separate files already classified as vowels and consonants
                trimmed_consonants = [ex[:length] for ex in consonant_examples]
                trimmed_vowels = [ex[:length] for ex in vowel_examples]
            else:
                # When the context length is less than 20, we need to trim the examples to `length + 1`
                # characters because we need to check the next character (context_length + 1 character)
                # to reclassify them as either vowels or consonants.
                trimmed_text = [ex[:length + 1] for ex in consonant_examples + vowel_examples]

                trimmed_vowels = []      # Initialize empty list for vowels
                trimmed_consonants = []  # Initialize empty list for consonants

                # Iterate through the trimmed examples and reclassify based on the last character.
                for ex in trimmed_text:
                    if ex[-1] in vowels:  # If the last character is a vowel, classify as vowel
                        trimmed_vowels.append(ex[:-1])  # Remove the last character and add to vowel list
                    else:  # If the last character isn't a vowel, classify as consonant
                        trimmed_consonants.append(ex[:-1])  # Remove the last character and add to consonant list

            # Append data to lists to analyze the class imbalances
            context_lengths.append(length)
            vowel_counts.append(len(trimmed_vowels))
            consonant_counts.append(len(trimmed_consonants))

            # Model evaluation: Calculate accuracy for the current context length
            correct_predictions = sum(1 for ex in trimmed_consonants if model.predict(ex) == 0)
            correct_predictions += sum(1 for ex in trimmed_vowels if model.predict(ex) == 1)
            total = len(trimmed_consonants) + len(trimmed_vowels)
            accuracy = correct_predictions / total

            accuracy_by_length[length] = accuracy
            if print_ok:
                print(f"Context length {length}: Accuracy = {accuracy:.3f}")
            else:
                pass

        # Prepare data for class imbalance visualization
        class_imbalances_data = {
            'context_length': context_lengths,
            'vowel_count': vowel_counts,
            'consonant_count': consonant_counts
        }

        return accuracy_by_length, class_imbalances_data

    def visualize_accuracy_trend(accuracy_data, title):
        """Visualize the accuracy trend across different context length and save the plot"""

        save_path = os.path.join('plots', 'accuracy_vs_context_length.png')
        lengths = list(accuracy_data.keys())
        accuracies = list(accuracy_data.values())

        plt.figure(figsize=(10, 6))
        plt.plot(lengths, accuracies, marker='o')
        plt.title(title)
        plt.xlabel("Context Length")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.savefig(save_path)  # Save the plot to the specified file path
        plt.close()
        print(f'Accuracy vs context length plot was saved to {save_path}.')

    def visualize_class_imbalance(data):
        """Visualize class imbalance between vowel and consonant counts 
        across different context lengths and save the plot"""

        save_path = os.path.join('plots', 'class_imbalance_vs_context_length.png')
        x = np.arange(len(data['context_length']))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width / 2, data['vowel_count'], width, label='Vowel Count', color='navy')
        bars2 = ax.bar(x + width / 2, data['consonant_count'], width, label='Consonant Count', color='orange')

        # Adding labels and title
        ax.set_xlabel('Context Length')
        ax.set_ylabel('Count')
        ax.set_title('Vowel and Consonant Counts by Context Length')
        ax.set_xticks(x)
        ax.set_xticklabels(data['context_length'])
        ax.legend()

        # Adding count labels on top of the bars
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        rotation=90)
        plt.savefig(save_path)  
        plt.close()
        print(f'Class imbalance vs context length plot was saved to {save_path}.')

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

    # Evaluate and visualize the results
    print("\nEvaluating RNN Classifier for different context lengths")
    rnn_accuracy_data, class_imbalance_data = evaluate_model_on_context_lengths(rnn_model, dev_cons_exs, dev_vowel_exs)
    visualize_accuracy_trend(rnn_accuracy_data, "RNN Classifier Accuracy vs. Context Length")
    visualize_class_imbalance(class_imbalance_data)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the consonant vowel classification experiment for different "
                                                 "context length.")
    # Add argument for the maximum context length, with a default of 20.
    parser.add_argument("--max_context_length", type=int, default=20, help="Maximum context length for evaluation.")
    parser.add_argument("--print", type=bool, default=True, help="Print evaluation results for each context length.")

    # Parse arguments
    args = parser.parse_args()

    # Run the experiment with parsed arguments
    run_experiment(args.max_context_length)


if __name__ == "__main__":
    main()


