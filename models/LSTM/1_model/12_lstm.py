
# original reference: https://github.com/CaptainE/RNN-LSTM-in-numpy
##########################################################################
##
##  PART 1
##
##########################################################################
import numpy as np
np.random.seed(42)
from collections import defaultdict
from torch.utils import data
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
def init_orthogonal(param):
    """    
    Initializes weight parameters orthogonally. (Meaning orthogonal matrix)
       Refer to this paper for an explanation of this initialization:
       https://arxiv.org/abs/1312.6120
    """
    
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
def init_lstm(hidden_size, vocab_size, z_size):
    """
    Initializes our LSTM network.
    Args:
     `hidden_size`: the dimensions of the hidden state
     `vocab_size`: the dimensions of our vocabulary
     `z_size`: the dimensions of the concatenated input 
    """
    # Weight matrix (forget gate)
    
    W_f = np.random.randn(hidden_size, z_size)
    
    # Bias for forget gate
    b_f = np.zeros((hidden_size, 1))

    # Weight matrix (input gate)
    W_i = np.random.randn(hidden_size, z_size)
    
    # Bias for input gate
    b_i = np.zeros((hidden_size, 1))

    # Weight matrix (candidate)
    W_g = np.random.randn(hidden_size, z_size)
    
    # Bias for candidate
    b_g = np.zeros((hidden_size, 1))

    # Weight matrix of the output gate
    W_o = np.random.randn(hidden_size, z_size)
    b_o = np.zeros((hidden_size, 1))

    # Weight matrix relating the hidden-state to the output
    W_v = np.random.randn(vocab_size, hidden_size)
    b_v = np.zeros((vocab_size, 1))
    
    # Initialize weights according to https://arxiv.org/abs/1312.6120
    W_f = init_orthogonal(W_f)
    W_i = init_orthogonal(W_i)
    W_g = init_orthogonal(W_g)
    W_o = init_orthogonal(W_o)
    W_v = init_orthogonal(W_v)

    return W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part1_I():
    print('Part 1 - I')
    sequences = generate_dataset()
    print('A single sample from the generated dataset:')
    print(sequences[0])
    return {
        'sequences': sequences
    }
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part1_II(sequences):
    print('Part 1 - II')
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
def execute_part1_III(sequences):
    print('Part 1 - III')
    training_set, validation_set, test_set = create_datasets(sequences = sequences, dataset_class = Dataset)
    return {
        'training_set': training_set,
        'validation_set': validation_set,
        'test_set': test_set
    }
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part1(hidden_layer_size = 50):
    print('PART 1')
    print('----------------')
    part1_I_result = execute_part1_I()
    print('----------------')
    part1_II_result = execute_part1_II(part1_I_result['sequences'])
    print('----------------')
    part1_III_result = execute_part1_III(part1_I_result['sequences'])
    print('----------------')
    
    vocab_size = part1_II_result['vocab_size']
    z_size = hidden_layer_size + vocab_size 

    params = init_lstm(hidden_size=hidden_layer_size, vocab_size=vocab_size, z_size=z_size)

    return {
        'sequences'     : part1_I_result['sequences'],
        'vocab_size'    : vocab_size,
        'word_to_idx'   : part1_II_result['word_to_idx'],
        'idx_to_word'   : part1_II_result['idx_to_word'],
        'num_sequences' : part1_II_result['num_sequences'],
        'hidden_layer_size': hidden_layer_size,
        'params'        : params,
        'training_set'  : part1_III_result['training_set'],
        'validation_set': part1_III_result['validation_set'],
        'test_set'      : part1_III_result['test_set']
    }
#-------------------------------------------------------------------------
part1_result = execute_part1()

##########################################################################
##
##  PART 2
##
##########################################################################
#-------------------------------------------------------------------------
def sigmoid(x, derivative = False):
    """    
    Computes the element-wise sigmoid activation function for an array x.
    
    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-12 # this is a low-low value
    f_x = 1 / (1 + np.exp(-x_safe))
    
    #sigmoid function: f(x) = 1 / (1 + e^(-x))
    #derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    if derivative: # Return the derivative of the function evaluated at x
        d_f_x = f_x * (1 - f_x)
        return d_f_x
    else: # Return the forward pass of the function at x
        return f_x
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def tanh(x, derivative = False):
    """
    Computes the element-wise tanh activation function for an array x.
    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass
    """
    #-----
    # tanh function: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))   ALSO: f(x) = sinh(x) / cosh(x)
    # derivative of tanh: f'(x) = 1 - f(x)^2                  OR f'(x) = 1 - tanh^2(x)

    x_safe = x + 1e-12 # this is a low-low value
    f_x = (np.exp(x_safe)-np.exp(-x_safe))/(np.exp(x_safe)+np.exp(-x_safe))
    
    if derivative: # Return the derivative of the function evaluated at x
        d_f_x = 1 - f_x**2
        return d_f_x
    else: # Return the forward pass of the function at x
        return f_x
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def softmax(x, derivative = False ):
    """
    Computes the softmax for an array x.
    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass\
    """
    #-----
    # softmax function: f(x) = e^x / sum(e^x)
    # derivative of softmax: "The derivative of the softmax function is a bit more complicated because softmax depends on all the elements of the input vector x ."

    x_safe = x + 1e-12 # this is a low-low value
    f_x = np.exp(x_safe) / np.sum(np.exp(x_safe))
    
    if derivative: # Return the derivative of the function evaluated at x
        pass # We will not need this one (truth is the derivative of softmax is very hard)
    else: # Return the forward pass of the function at x
        return f_x
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
def forward(inputs, h_prev, C_prev, p, hidden_size):
    """
    Arguments:
    x -- your input data at timestep "t", numpy array of shape (n_x, m).
    h_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    C_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    p -- python list containing:
                        W_f -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        b_f -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        W_i -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        b_i -- Bias of the update gate, numpy array of shape (n_a, 1)
                        W_g -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        b_g --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        W_o -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        b_o --  Bias of the output gate, numpy array of shape (n_a, 1)
                        W_v -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_v, n_a)
                        b_v -- Bias relating the hidden-state to the output, numpy array of shape (n_v, 1)
    Returns:
    z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s -- lists of size m containing the computations in each forward pass
    outputs -- prediction at timestep "t", numpy array of shape (n_v, m)
    """
    assert h_prev.shape == (hidden_size, 1)
    assert C_prev.shape == (hidden_size, 1)

    # First we unpack our parameters
    W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = p
    
    # Save a list of computations for each of the components in the LSTM
    x_s, z_s, f_s, i_s,  = [], [] ,[], []
    g_s, C_s, o_s, h_s = [], [] ,[], []
    v_s, output_s =  [], [] 
    
    # Append the initial cell and hidden state to their respective lists
    h_s.append(h_prev)
    C_s.append(C_prev)
    
    for x in inputs:
        
        # YOUR CODE HERE!
        # Concatenate input and hidden state
        z = np.row_stack((h_prev, x))
        z_s.append(z)
        
        # YOUR CODE HERE!
        # Calculate forget gate
        f = sigmoid(np.dot(W_f, z) + b_f)
        f_s.append(f)
        
        # Calculate input gate
        i = sigmoid(np.dot(W_i, z) + b_i)
        i_s.append(i)
        
        # Calculate candidate
        g = tanh(np.dot(W_g, z) + b_g)
        g_s.append(g)
        
        # YOUR CODE HERE!
        # Calculate memory state
        C_prev = f * C_prev + i * g 
        C_s.append(C_prev)
        
        # Calculate output gate
        o = sigmoid(np.dot(W_o, z) + b_o)
        o_s.append(o)
        
        # Calculate hidden state
        h_prev = o * tanh(C_prev)
        h_s.append(h_prev)

        # Calculate logits
        v = np.dot(W_v, h_prev) + b_v
        v_s.append(v)
        
        # Calculate softmax
        output = softmax(v)
        output_s.append(output)

    return z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, output_s
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part2(vocab_size, hidden_layer_size, params, idx_to_word, word_to_idx, test_set):
    print('PART 2')
    # Get first sentence in test set
    inputs, targets = test_set[1]

    # One-hot encode input and target sequence
    inputs_one_hot = one_hot_encode_sequence(sequence = inputs, vocab_size = vocab_size, word_to_idx=word_to_idx)
    targets_one_hot = one_hot_encode_sequence(sequence = targets, vocab_size = vocab_size, word_to_idx=word_to_idx)

    # Initialize hidden state as zeros
    h = np.zeros((hidden_layer_size, 1))
    c = np.zeros((hidden_layer_size, 1))

    # Forward pass
    z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward(inputs_one_hot, h, c, params, hidden_layer_size)

    output_sentence = [idx_to_word[np.argmax(output)] for output in outputs]
    print('Input sentence:')
    print(inputs)

    print('\nTarget sequence:')
    print(targets)

    print('\nPredicted sequence:')
    print([idx_to_word[np.argmax(output)] for output in outputs])
#-------------------------------------------------------------------------

execute_part2(
    vocab_size          = part1_result['vocab_size'],
    hidden_layer_size   = part1_result['hidden_layer_size'],
    params              = part1_result['params'],
    idx_to_word         = part1_result['idx_to_word'],
    word_to_idx         = part1_result['word_to_idx'],
    test_set            = part1_result['test_set']
)
