import numpy as np
# original reference: https://github.com/CaptainE/RNN-LSTM-in-numpy
##########################################################################
##
##  PART 1
##
##########################################################################
# Set seed such that we always get the same dataset
np.random.seed(42)
#-------------------------------------------------------------------------
def generate_dataset(num_sequences=100):

    # Generates a number of sequences as our dataset.
    #
    # Args:
    # `num_sequences`: the number of sequences to be generated.
    #     
    #Returns a list of sequences.

    samples = []
    
    for _ in range(num_sequences):  
        num_tokens = np.random.randint(1, 10) # pick a number of tokens to be generated
        # generate an equal number of 'a's and 'b's, followed by an 'EOS' token, e.g.: ['a', 'a', 'a', 'b', 'b', 'b', 'EOS']
        sample = ['a'] * num_tokens + ['b'] * num_tokens + ['EOS']
        samples.append(sample)
        
    return samples
#-------------------------------------------------------------------------
print('Generating dataset of sequences')
sequences = generate_dataset()

print('A single sample from the generated dataset:')
print(sequences[0])
print('----------------------------------------------')

##########################################################################
##
##  PART 2
##
##########################################################################
from collections import defaultdict
#-------------------------------------------------------------------------
def sequences_to_dicts(sequences):
    # Creates word_to_idx and idx_to_word dictionaries for a list of sequences.
    
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
    word_to_idx = defaultdict(lambda: num_words) #probably should be vocab_size
    idx_to_word = defaultdict(lambda: 'UNK')

    # Fill dictionaries
    for idx, word in enumerate(unique_words):
        # YOUR CODE HERE!
        word_to_idx[word] = idx
        idx_to_word[idx] = word

    return word_to_idx, idx_to_word, num_sentences, vocab_size
#-------------------------------------------------------------------------

##########################################################################
##
##  PART 3
##
##########################################################################
from torch.utils import data
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
def create_datasets(sequences, dataset_class, p_train=0.8, p_val=0.1, p_test=0.1):
    # Define partition sizes
    num_train = int( len(sequences)*p_train ) #typically 80% of the data
    num_validation = int( len(sequences)*p_val )     #typically 10% of the data
    num_test = int( len(sequences)*p_test )   #typically 10% of the data

    # Split sequences into partitions
    sequences_train = sequences[:num_train]
    sequences_validation = sequences[num_train:num_train+num_validation]
    sequences_test = sequences[-num_test:]

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

training_set, validation_set, test_set = create_datasets(sequences = sequences, dataset_class = Dataset)

print(f'We have {len(training_set)} samples in the training set.')
print(f'We have {len(validation_set)} samples in the validation set.')
print(f'We have {len(test_set)} samples in the test set.')


word_to_idx, idx_to_word, num_sequences, vocab_size = sequences_to_dicts(sequences = sequences)

print(f'We have {num_sequences} sentences and {len(word_to_idx)} unique tokens in our dataset (including UNK).\n')
print('The index of \'b\' is', word_to_idx['b'])
print(f'The word corresponding to index 1 is \'{idx_to_word[1]}\'')


##########################################################################
##
##  PART 4
##
##########################################################################

#-------------------------------------------------------------------------    
def one_hot_encode(idx, vocab_size):
    # One-hot encodes a single word given its index and the size of the vocabulary.
    #    
    # Args:
    #  `idx`: the index of the given word
    #  `vocab_size`: the size of the vocabulary
    #
    # Returns a 1-D numpy array of length `vocab_size`.

    # Initialize the encoded array
    one_hot = np.zeros(vocab_size)
    
    # Set the appropriate element to one
    one_hot[idx] = 1.0

    return one_hot
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------    
def one_hot_encode_sequence(sequence, vocab_size, param_word_to_idx):

    # One-hot encodes a sequence of words given a fixed vocabulary size.
    #
    # Args:
    #  `sentence`: a list of words to encode
    #  `vocab_size`: the size of the vocabulary
    #     
    # Returns a 3-D numpy array of shape (num words, vocab size, 1).

    # Encode each word in the sentence. From each word in a sentence, we send the index of the word and vocab_size
    #  we expect to receive back the one-hot encoding of the word, something like [0,0,0,1,0,0,0,...]
    encoding = np.array(
            [   
                one_hot_encode(param_word_to_idx[word], vocab_size) 
                for word in sequence
            ]
        )

    # Reshape encoding to (len_num words, len_vocab size, 1)
    encoding = encoding.reshape( encoding.shape[0], encoding.shape[1], 1 )
    
    return encoding
#-------------------------------------------------------------------------    

test_word = one_hot_encode(idx = word_to_idx['a'], vocab_size = vocab_size)
print(f'Our one-hot encoding of \'a\' has shape {test_word.shape}.')

test_sentence = one_hot_encode_sequence(sequence = ['a', 'b'], vocab_size = vocab_size, param_word_to_idx = word_to_idx)
print(f'Our one-hot encoding of \'a b\' has shape {test_sentence.shape}.')

##########################################################################
##
##  PART 5
##
##########################################################################


hidden_size = 50 # Number of dimensions in the hidden state
vocab_size  = len(word_to_idx) # Size of the vocabulary used

#-------------------------------------------------------------------------  
def init_orthogonal(param):
    
    # Initializes weight parameters orthogonally. (Meaning orthogonal matrix)
    #    Refer to this paper for an explanation of this initialization:
    #    https://arxiv.org/abs/1312.6120
    
    if param.ndim < 2:
        raise ValueError("Only parameters with 2 or more dimensions are supported.")

    rows, cols = param.shape
    
    new_param = np.random.randn(rows, cols) #randn -> sample(s) standard normal distribution in the shape of rows x cols
    
    if rows < cols:
        new_param = new_param.T #transpose the matrix, if 3x2 -> 2x3
    
    # Compute QR factorization
    q, r = np.linalg.qr(new_param)
    
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    # consider:
    # r = np.array([[0, 1, 2],
    #               [3, 4, 5],
    #               [6, 7, 8]])
    # np.diag(r, 0) -> [0 4 8]    np.diag(r, 1) -> [1 5]    np.diag(r, -1) -> [3 7]
    d = np.diag(r, 0)

    # np.sign(d): This function returns an array of the same shape as d, where each element is the sign of the corresponding element in d.
    #   If an element in d is positive, the corresponding element in ph will be 1.
    #   If an element in d is negative, the corresponding element in ph will be -1.
    #   If an element in d is zero, the corresponding element in ph will be 0.
    # Consider: d = np.array([4, -5, 0]) -> ph = np.sign(d) -> [1 -1 0]
    ph = np.sign(d)

    q *= ph # multiplies each column of q by each element in ph, ensuring that the orthogonal matrix q has the desired properties

    if rows < cols:
        q = q.T
    
    new_param = q
    
    return new_param
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def init_rnn(hidden_size, vocab_size):

    # Initializes our recurrent neural network.
    #
    # Args:
    #  `hidden_size`: the dimensions of the hidden state
    #  `vocab_size`: the dimensions of our vocabulary

    # Weight matrix (input to hidden state)
    U = np.zeros((hidden_size, vocab_size))

    # Weight matrix (recurrent computation)
    V = np.zeros((hidden_size, hidden_size))

    # Weight matrix (hidden state to output)
    W = np.zeros((vocab_size, hidden_size))

    # Bias (hidden state)
    b_hidden = np.zeros((hidden_size, 1))

    # Bias (output)
    b_out = np.zeros((vocab_size, 1))
    
    # Initialize weights
    U = init_orthogonal(U)
    V = init_orthogonal(V)
    W = init_orthogonal(W)
    
    # Return parameters as a tuple
    return U, V, W, b_hidden, b_out
#-------------------------------------------------------------------------

params = init_rnn(hidden_size=hidden_size, vocab_size=vocab_size)
print('----------------------------------------------')
print(f'U (weight input to hidden state) shape: {params[0].shape}') # U
print(f'V (weight matrix recurrent computation) shape: {params[1].shape}') # V
print(f'W (weight matrix hidden state to output) shape: {params[2].shape}') # W
print(f'bias_hidden shape: {params[3].shape}') # b_hidden
print(f'bias_out shape: {params[4].shape}') # b_out
print('----------------------------------------------')
# print(f'U first 5 rows:\n------------\n{params[0][:5]}\n------------\n')
# print(f'V first 2 rows:\n------------\n{params[1][:2]}\n------------\n')
# print(f'W first 5 rows:\n------------\n{params[2][:2]}\n------------\n')
# print(f'bias_hidden first 5 rows:{params[3]}')
# print(f'bias_out :{params[4]}')

##########################################################################
##
##  PART 6
##
##########################################################################

#-------------------------------------------------------------------------
def sigmoid(x, derivative = False):
    
    # Computes the element-wise sigmoid activation function for an array x.
    #
    # Args:
    #  `x`: the array where the function is applied
    #  `derivative`: if set to True will return the derivative instead of the forward pass
    
    x_safe = x + 1e-12 # this is a low-low value
    f = 1 / (1 + np.exp(-x_safe))
    
    #sigmoid function: f(x) = 1 / (1 + e^(-x))
    #derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    if derivative: # Return the derivative of the function evaluated at x
        return f * (1 - f)
    else: # Return the forward pass of the function at x
        return f
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def tanh(x, derivative = False):

    # Computes the element-wise tanh activation function for an array x.

    # Args:
    #  `x`: the array where the function is applied
    #  `derivative`: if set to True will return the derivative instead of the forward pass
    #-----
    # tanh function: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))   ALSO: f(x) = sinh(x) / cosh(x)
    # derivative of tanh: f'(x) = 1 - f(x)^2                  OR f'(x) = 1 - tanh^2(x)

    x_safe = x + 1e-12 # this is a low-low value
    f = (np.exp(x_safe)-np.exp(-x_safe))/(np.exp(x_safe)+np.exp(-x_safe))
    
    if derivative: # Return the derivative of the function evaluated at x
        return 1 - f**2
    else: # Return the forward pass of the function at x
        return f
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def softmax(x, derivative = False ):

    # Computes the softmax for an array x.
    #
    # Args:
    #  `x`: the array where the function is applied
    #  `derivative`: if set to True will return the derivative instead of the forward pass
    #-----
    # softmax function: f(x) = e^x / sum(e^x)
    # derivative of softmax: "The derivative of the softmax function is a bit more complicated because softmax depends on all the elements of the input vector x ."

    x_safe = x + 1e-12 # this is a low-low value
    f = np.exp(x_safe) / np.sum(np.exp(x_safe))
    
    if derivative: # Return the derivative of the function evaluated at x
        pass # We will not need this one
    else: # Return the forward pass of the function at x
        return f
#-------------------------------------------------------------------------