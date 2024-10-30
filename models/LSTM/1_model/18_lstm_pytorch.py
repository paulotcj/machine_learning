import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from torch.utils import data

np.random.seed(42)


##########################################################################
##
##  PART 15
##
##########################################################################

#-------------------------------------------------------------------------
class Net(nn.Module):
    #-------------------------------------------------------------------------
    def __init__(self, vocab_size, hidden_layer_size = 50, num_layers = 1):
        super(Net, self).__init__()

        # Recurrent layer
        self.lstm = nn.LSTM(
            input_size     = vocab_size, # typically 4 in our example
            hidden_size    = hidden_layer_size, # hidden layer size (LSTM units in this layer)
            num_layers     = num_layers,
            bidirectional  = False
        )
        
        # Output layer
        self.l_out = nn.Linear(
            in_features     = hidden_layer_size, # we are going to output a size 50 vector from the LSTM layer, that has 50 units
            out_features    = vocab_size,
            bias            = False
        )
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, x):
        # RNN returns output and last hidden state
        x, (h, c) = self.lstm(x) # x = input, h = hidden state, c = cell state
        
        # Flatten output for feed-forward layer
        #   -1: tells PyTorch to infer the size of this dimension based on the other dimensions and the total number of elements in the tensor. 
        #     Essentially, it allows PyTorch to automatically calculate the appropriate size for this dimension
        x = x.view(-1, self.lstm.hidden_size)
        
        # Output layer
        x = self.l_out(x)
        
        return x
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------

##########################################################################
##
##  PART 16
##
##########################################################################

import matplotlib.pyplot as plt
#-------------------------------------------------------------------------
def generate_dataset(num_sequences=100):
    """
    Generates a number of sequences as our dataset.
    Args:
     `num_sequences`: the number of sequences to be generated.
    Returns a list of sequences.
    """

    samples = []
    
    for _ in range(num_sequences):  
        num_tokens = np.random.randint(1, 10) # pick a number of tokens to be generated
        # generate an equal number of 'a's and 'b's, followed by an 'EOS' token, e.g.: ['a', 'a', 'a', 'b', 'b', 'b', 'EOS']
        sample = ['a'] * num_tokens + ['b'] * num_tokens + ['EOS']
        samples.append(sample)
        
    return samples
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def sequences_to_dicts(sequences):
    """
    Creates word_to_idx and idx_to_word dictionaries for a list of sequences.
    Args:
        `sequences`: a list of sequences
    """
    
    # A bit of Python-magic to flatten a nested list 'list_param'
    flatten = lambda list_param: [
        item 
        for sublist in list_param #loop over the elements of list_param creating a sublist
        for item in sublist #loop over the elements of sublist creating an item
    ]
    
    # Flatten the dataset
    all_words = flatten(sequences)
    
    # Count number of word occurences
    word_count = defaultdict(int) #defaultdict is a dictionary that returns a default value if the key is not found, here the default value is 0
    for word in flatten(sequences):
        word_count[word] += 1

    # Sort by frequency in descending order
    # word_count.items() returns a list of (word, count) pairs. 'l' is a tuple (word, count). Then we sort the list of tuples by the count '-l[1]'
    # this would be better as: word_count_desc = sorted(list(word_count.items()), key=lambda tuple_item: tuple_item[1], reverse=True)
    word_count_desc = sorted(list(word_count.items()), key=lambda l: -l[1])

    # Create a list of all unique words
    unique_words = [item[0] for item in word_count_desc]
    
    # Add UNK token to list of words
    unique_words.append('UNK')

    # Count number of sequences and number of unique words
    num_sentences, vocab_size = len(sequences), len(unique_words)

    # Create dictionaries so that we can go from word to index and back
    # If a word is not in our vocabulary, we assign it to token 'UNK'
    # word_to_idx = defaultdict(lambda: num_words) #probably should be vocab_size
    word_to_idx = defaultdict(lambda: vocab_size)
    idx_to_word = defaultdict(lambda: 'UNK')

    # Fill dictionaries
    for idx, word in enumerate(unique_words):
        word_to_idx[word] = idx
        idx_to_word[idx] = word

    return word_to_idx, idx_to_word, num_sentences, vocab_size
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class Dataset(data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs # x
        self.targets = targets # y

    def __len__(self):
        # Return the size of the dataset
        return len(self.targets)

    def __getitem__(self, index):
        # Retrieve inputs and targets at the given index
        x = self.inputs[index]
        y = self.targets[index]

        return x, y
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def get_inputs_targets_from_sequences(sequences):
    # Define empty lists
    inputs, targets = [], []
    
    # Append inputs and targets such that both lists contain L-1 words of a sentence of length L
    # but targets are shifted right by one so that we can predict the next word
    # example: 'The quick brown fox jumps'
    #    Inputs:  ["The", "quick", "brown", "fox"]
    #    Targets: ["quick", "brown", "fox", "jumps"]
    # The idea is to predict the next word based on the input words, so the model learns to predict 
    # "quick" given "The", "brown" given "quick", and so on.
    for sequence in sequences:
        inputs.append(sequence[:-1]) # take everything except the last word -> ["The", "quick", "brown", "fox"]
        targets.append(sequence[1:]) # take everything except the first word -> ["quick", "brown", "fox", "jumps"]
        
    return inputs, targets
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def create_datasets(sequences, dataset_class, p_train=0.8, p_val=0.1, p_test=0.1):
    # Define partition sizes
    num_train = int( len(sequences)*p_train ) #typically 80% of the data
    num_validation = int( len(sequences)*p_val )     #typically 10% of the data
    num_test = int( len(sequences)*p_test )   #typically 10% of the data

    # Split sequences into partitions
    sequences_train = sequences[:num_train]
    sequences_validation = sequences[num_train:num_train+num_validation]
    sequences_test = sequences[-num_test:]

    # In the next step we split this into inputs and targets. So far we have sequences for: 
    #   Training, Validation, and Test. After that we will have:
    #   Training (Inputs, Targets), Validation (Inputs, Targets), and Test (Inputs, Targets)

    # Get inputs and targets for each partition
    inputs_train, targets_train = get_inputs_targets_from_sequences(sequences_train)
    inputs_validation, targets_validation = get_inputs_targets_from_sequences(sequences_validation)
    inputs_test, targets_test = get_inputs_targets_from_sequences(sequences_test)

    # Create datasets
    training_set = dataset_class(inputs_train, targets_train)
    validation_set = dataset_class(inputs_validation, targets_validation)
    test_set = dataset_class(inputs_test, targets_test)

    return training_set, validation_set, test_set
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------    
def one_hot_encode(idx, vocab_size):
    """
    One-hot encodes a single word given its index and the size of the vocabulary.
       
    Args:
     `idx`: the index of the given word
     `vocab_size`: the size of the vocabulary
    
    Returns a 1-D numpy array of length `vocab_size`.
    """

    # Initialize the encoded array
    one_hot = np.zeros(vocab_size)
    
    # Set the appropriate element to one
    one_hot[idx] = 1.0

    return one_hot
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------    
def one_hot_encode_sequence(sequence, vocab_size, word_to_idx):
    """
    One-hot encodes a sequence of words given a fixed vocabulary size.
    
    Args:
     `sentence`: a list of words to encode
     `vocab_size`: the size of the vocabulary
        
    Returns a 3-D numpy array of shape (num words, vocab size, 1).
    """

    # Encode each word in the sentence. From each word in a sentence, we send the index of the word and vocab_size
    #  we expect to receive back the one-hot encoding of the word, something like [0,0,0,1,0,0,0,...]
    encoding = np.array(
            [   
                one_hot_encode(word_to_idx[word], vocab_size) 
                for word in sequence
            ]
        )

    # Reshape encoding to (len_num words, len_vocab size, 1)
    encoding = encoding.reshape( encoding.shape[0], encoding.shape[1], 1 )
    
    return encoding
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part11_I():
    print('Part 11 - I')
    sequences = generate_dataset()
    print('A single sample from the generated dataset:')
    print(sequences[0])
    return {
        'sequences': sequences
    }
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part11_II(sequences):
    print('Part 11 - II')
    word_to_idx, idx_to_word, num_sequences, vocab_size = sequences_to_dicts(sequences)
    print(f'We have {num_sequences} sentences and {len(word_to_idx)} unique tokens in our dataset (including UNK).\n')
    print('The index of \'b\' is', word_to_idx['b'])
    print(f'The word corresponding to index 1 is \'{idx_to_word[1]}\'')

    print(f'vocab_size: {vocab_size}')
    for i in range(vocab_size):
        print(f'{i}: {idx_to_word[i]}')      
    return {
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,
        'num_sequences': num_sequences,
        'vocab_size': vocab_size
    }
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part11_III(sequences):
    print('Part 11 - III')
    training_set, validation_set, test_set = create_datasets(sequences = sequences, dataset_class = Dataset)
    return {
        'training_set': training_set,
        'validation_set': validation_set,
        'test_set': test_set
    }
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part11(hidden_layer_size = 50):
    print('PART 11')
    print('----------------')
    part11_I_result = execute_part11_I()
    print('----------------')
    part11_II_result = execute_part11_II(part11_I_result['sequences'])
    print('----------------')
    part11_III_result = execute_part11_III(part11_I_result['sequences'])
    print('----------------')
    
    vocab_size = part11_II_result['vocab_size']
    # z_size = hidden_layer_size + vocab_size # typically: 50 + 4 = 54
    # params = init_lstm(hidden_layer_size=hidden_layer_size, vocab_size=vocab_size, z_size=z_size)

    return {
        'sequences'     : part11_I_result['sequences'],
        'vocab_size'    : vocab_size,
        'word_to_idx'   : part11_II_result['word_to_idx'],
        'idx_to_word'   : part11_II_result['idx_to_word'],
        'num_sequences' : part11_II_result['num_sequences'],
        'hidden_layer_size': hidden_layer_size,
        # 'params'        : params,
        'training_set'  : part11_III_result['training_set'],
        'validation_set': part11_III_result['validation_set'],
        'test_set'      : part11_III_result['test_set']
    }
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def train_lstm(validation_set, training_set, word_to_idx, vocab_size):
    # Hyper-parameters
    num_epochs = 1

    # Initialize a new network
    net = Net(vocab_size = vocab_size)

    # Define a loss function and optimizer for this problem
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

    # Track loss
    training_loss, validation_loss = [], []

    #-------------------------------------------------
    # For each epoch
    for i in range(num_epochs):
        
        # Track loss
        epoch_training_loss = 0
        epoch_validation_loss = 0
        
        #----------
        # net.eval() is a method call in PyTorch that sets the neural network model net to evaluation mode. 
        #   This is particularly important when you are transitioning from the training phase to the 
        #   evaluation or inference phase of your model's lifecycle. During training, certain layers 
        #   like Dropout and BatchNorm behave differently to facilitate learning. For instance, Dropout 
        #   randomly zeroes some of the elements of the input tensor with a given probability during 
        #   training to prevent overfitting. BatchNorm normalizes the output of a previous activation 
        #   layer by subtracting the batch mean and dividing by the batch standard deviation.
        #
        # When you call net.eval(), it changes the behavior of these layers to be appropriate for 
        #   evaluation. Specifically, Dropout layers will no longer drop any units, and BatchNorm layers 
        #   will use the learned running statistics instead of batch statistics. This ensures that the 
        #   model's predictions are deterministic and consistent, which is crucial for evaluating the 
        #   model's performance on a validation set or making predictions on new data.        
        net.eval()
        #----------
            
        #-------------------------------------------------
        # For each sentence in validation set
        for inputs, targets in validation_set:
            
            # One-hot encode input and target sequence
            inputs_one_hot = one_hot_encode_sequence(sequence = inputs, vocab_size=vocab_size, word_to_idx=word_to_idx)
            
            
            # consider a target sequence: [a, a, b, b, b, EOS], and the word_to_idx represents 
            #   a - 0, b - 1, EOS - 2, UNK - 3. That means that the targets_idx will be [0, 0, 1, 1, 1, 2]
            targets_idx = [ word_to_idx[word] for word in targets ]

            
            #----------------
            # Convert input to tensor
            #  lets break this down: the first input las a length of 6, our vocabulary has a length of 4
            #  therefore input is len 6 -> ['a', 'a', 'a', 'b', 'b', 'b'], and input_one_hot is len 6x4x1 ->
            #     [ [ [1.], [0.], [0.], [0.] ], ... ]
            #  but then inputs_one_hot will be wrapped in a tensor object -> torch.Size([6, 4, 1])
            inputs_one_hot = torch.Tensor(inputs_one_hot)

            #-----------------
            # inputs_one_hot is being rearranged such that the first dimension (index 0) remains in place, 
            #   the second dimension (index 1) is moved to the third position, and the third dimension 
            #   (index 2) is moved to the second position. It was [6,4,1] and it will become [6,1,4].
            #
            # To break it down further, if inputs_one_hot originally had dimensions 
            #   [batch_size, seq_len, num_features], after the permutation, it would have dimensions 
            #   [batch_size, num_features, seq_len]
            inputs_one_hot = inputs_one_hot.permute(0, 2, 1)



            # Convert target to tensor
            targets_idx = torch.LongTensor(targets_idx) #just wraps the list in a tensor object

            
            # Forward pass
            #  considering a target sentence of length 6 and vocab len 4 the shape will be: [6, 4]
            outputs = net.forward(inputs_one_hot) 
            
            # Compute loss
            loss = criterion(outputs, targets_idx)
            
            # Update loss
            epoch_validation_loss += loss.detach().numpy()
        #-------------------------------------------------

        # When a model is in training mode, certain layers like Dropout and BatchNorm behave differently 
        #  compared to when the model is in evaluation mode. Specifically, Dropout layers will randomly 
        #  zero some of the elements of the input tensor with a given probability, which helps prevent 
        #  overfitting. BatchNorm layers will use the statistics of the current batch rather than the 
        #  running statistics accumulated during training.
        #
        # By calling net.train(), you are preparing the model net to be trained, enabling the specific 
        #   behaviors of certain layers that are necessary for effective training. This step is typically 
        #   followed by the actual training loop, where the model parameters are updated based on the 
        #   loss computed from the training data.
        net.train()
        
        #-------------------------------------------------
        # For each sentence in training set
        for inputs, targets in training_set:
            
            # One-hot encode input and target sequence
            inputs_one_hot = one_hot_encode_sequence(sequence = inputs, vocab_size = vocab_size, word_to_idx = word_to_idx)
            targets_idx = [word_to_idx[word] for word in targets] #see explanation above
            
            # Convert input to tensor
            inputs_one_hot = torch.Tensor(inputs_one_hot)
            inputs_one_hot = inputs_one_hot.permute(0, 2, 1) #see explanation above
            
            # Convert target to tensor
            targets_idx = torch.LongTensor(targets_idx)
            
            # Forward pass
            outputs = net.forward(inputs_one_hot)
            
            # Compute loss
            loss = criterion(outputs, targets_idx)
            
            # Backward pass
            #  PyTorch accumulates gradients by default. This means that if you don't reset the gradients, 
            #    they will be summed up across multiple backward passes, leading to incorrect updates.
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            
            # Update loss
            epoch_training_loss += loss.detach().numpy()
        #-------------------------------------------------

        # Save loss for plot
        training_loss.append(epoch_training_loss/len(training_set))
        validation_loss.append(epoch_validation_loss/len(validation_set))

        # Print loss every 5 epochs
        if i % 5 == 0:
            print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')
    #-------------------------------------------------

    return {
        'net': net,
        'training_loss': training_loss,
        'validation_loss': validation_loss
    }
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def make_prediction(net, test_set, word_to_idx, idx_to_word, vocab_size):
            
    # Get first sentence in test set
    inputs, targets = test_set[1]

    # One-hot encode input and target sequence
    inputs_one_hot = one_hot_encode_sequence(sequence = inputs, vocab_size = vocab_size, word_to_idx=word_to_idx)
    targets_idx = [word_to_idx[word] for word in targets]

    # Convert input to tensor
    inputs_one_hot = torch.Tensor(inputs_one_hot)
    inputs_one_hot = inputs_one_hot.permute(0, 2, 1)

    # Convert target to tensor
    targets_idx = torch.LongTensor(targets_idx)

    # Forward pass
    outputs = net.forward(inputs_one_hot).data.numpy()

    print('\nInput sequence:')
    print(inputs)

    print('\nTarget sequence:')
    print(targets)

    print('\nPredicted sequence:')
    print([idx_to_word[np.argmax(output)] for output in outputs])
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def plot_graph(training_loss, validation_loss):
    # Plot training and validation loss
    epoch = np.arange(len(training_loss))
    plt.figure()
    plt.plot(epoch, training_loss, 'r', label='Training loss',)
    plt.plot(epoch, validation_loss, 'b', label='Validation loss')
    plt.legend()
    plt.xlabel('Epoch'), plt.ylabel('NLL')
    plt.show()    
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part16():
    part11_result = execute_part11()
    train_lstm_result   = train_lstm(
        validation_set  = part11_result['validation_set'],
        training_set    = part11_result['training_set'],
        word_to_idx     = part11_result['word_to_idx'],
        vocab_size      = part11_result['vocab_size']
    )

    # make_prediction(
    make_prediction(
        net         = train_lstm_result['net'],
        test_set    = part11_result['test_set'], 
        word_to_idx = part11_result['word_to_idx'],
        idx_to_word = part11_result['idx_to_word'],
        vocab_size  = part11_result['vocab_size']
    )
    plot_graph(
        training_loss   = train_lstm_result['training_loss'], 
        validation_loss = train_lstm_result['validation_loss']
    )
#-------------------------------------------------------------------------
execute_part16()