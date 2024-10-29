
# original reference: https://github.com/CaptainE/RNN-LSTM-in-numpy
##########################################################################
##
##  PART 11: LSTM
##
##########################################################################
import numpy as np
from collections import defaultdict
from torch.utils import data
np.random.seed(42)
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
def init_lstm(hidden_layer_size, vocab_size, z_size):
    """
    Initializes our LSTM network.
    Args:
     `hidden_size`: the dimensions of the hidden state
     `vocab_size`: the dimensions of our vocabulary
     `z_size`: the dimensions of the concatenated input 
    """
    # print(f'z_size: {z_size}') # typically: 50 + 4 = 54
    #------------------
    # Standard init, weights with random and biases with zeros
    W_f = np.random.randn(hidden_layer_size, z_size) #weight matrix of the forget gate
    b_f = np.zeros((hidden_layer_size, 1)) #fotget gate bias
    #------
    W_i = np.random.randn(hidden_layer_size, z_size) #weight matrix of the input gate
    b_i = np.zeros((hidden_layer_size, 1)) #input gate bias
    #------
    # the candidate refers to the potential new values that could be added to the cell state
    W_g = np.random.randn(hidden_layer_size, z_size) #weight matrix of the candidate
    b_g = np.zeros((hidden_layer_size, 1)) #candidate bias
    #------
    W_o = np.random.randn(hidden_layer_size, z_size) #weight matrix of the output gate
    b_o = np.zeros((hidden_layer_size, 1)) #output bias
    #------
    W_v = np.random.randn(vocab_size, hidden_layer_size) #weight matrix relating the hidden-state to the output
    b_v = np.zeros((vocab_size, 1)) #bias relating the hidden-state to the output
    #------------------
    # Orthogonal init, for more details: https://arxiv.org/abs/1312.6120
    W_f = init_orthogonal(W_f)
    W_i = init_orthogonal(W_i)
    W_g = init_orthogonal(W_g)
    W_o = init_orthogonal(W_o)
    W_v = init_orthogonal(W_v)
    #------------------

    return W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v
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
    z_size = hidden_layer_size + vocab_size # typically: 50 + 4 = 54

    params = init_lstm(hidden_layer_size=hidden_layer_size, vocab_size=vocab_size, z_size=z_size)

    return {
        'sequences'     : part11_I_result['sequences'],
        'vocab_size'    : vocab_size,
        'word_to_idx'   : part11_II_result['word_to_idx'],
        'idx_to_word'   : part11_II_result['idx_to_word'],
        'num_sequences' : part11_II_result['num_sequences'],
        'hidden_layer_size': hidden_layer_size,
        'params'        : params,
        'training_set'  : part11_III_result['training_set'],
        'validation_set': part11_III_result['validation_set'],
        'test_set'      : part11_III_result['test_set']
    }
#-------------------------------------------------------------------------
part11_result = execute_part11()
##########################################################################
##
##  PART 12
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
def forward(inputs, hidden_state_prev, C_prev, params, hidden_layer_size):
    """
    Arguments:
    inputs -- your input data at timestep "t", numpy array of shape (n_x, m).
    hidden_state_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    C_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    params -- python list containing :
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
    Note: W_f and other weights have their size defined by z_size defined at init_lstm
    Returns:
    z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s -- lists of size m containing the computations in each forward pass
    outputs -- prediction at timestep "t", numpy array of shape (n_v, m)
    """
    # z_s will be a list of the concatenated input and hidden state
    # f_s will be a list of the forget gate computations
    # i_s will be a list of the input gate computations
    # g_s will be a list of the candidate computations
    # C_s will be a list of the cell state computations
    # o_s will be a list of the output gate computations
    # h_s will be a list of the hidden state computations
    # v_s will be a list of the logit computations

    assert hidden_state_prev.shape == (hidden_layer_size, 1)
    assert C_prev.shape == (hidden_layer_size, 1)
    #--------
    # First we unpack our parameters
    W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = params
    
    # Save a list of computations for each of the components in the LSTM
    x_s, z_s, f_s, i_s,  = [], [] ,[], []
    g_s, C_s, o_s, h_s = [], [] ,[], []
    v_s, output_s =  [], [] 
    #--------


    # Append the initial cell and hidden state to their respective lists
    h_s.append(hidden_state_prev) # 0's in the first execution
    C_s.append(C_prev) #0's in the first execution
    

    # inputs will be a list of one-hot encoded words, in our example most likely a sentence with 14 words, and the vocab size is 4,
    #  so the shape will be (14,4,1), and they will typically look like this: [[[1],[0],[0],[0]], [[0],[1],[0],[0]], ...]
    # and x will be [[1],[0],[0],[0]]
    for key, x in enumerate(inputs):
        # Concatenate input and hidden state
        z = np.row_stack((hidden_state_prev, x)) #this one will look like (54,1) and [[0],[1],[0],[0],...,[1],[0],[0],[0]] - where this last [1],[0],[0],[0] is x
        z_s.append(z) 

        #--------
        # Notes and refreshers: Dot product - Consider matrices A and B. The dot product of A and B requires thart number of columns in A to be
        #  equal to the number of rows in B. And the result will be the number of rows in A and the number of columns in B.
        #  For example: A shape (50,54) and B shape (54,1) will result in a matrix of shape (50,1)
        #                        R   C               R  C  -> A_cols = 54, B_rows = 54, A_rows = 50, B_cols = 1, result = (50,1)
        #--------
        # Notes on addition of 2 np.arrays: Considering the shape we are dealing with (50,1), adding 2 arrays the result is performed element-wise,
        #  for instance: [[1],[2],[3]] + [[4],[5],[6]] = [[5],[7],[9]]
        #--------

        # Calculate forget gate, W_f is the weights for the forget gate, tipycally in our example, 50 layers of 54 z's
        # W_f shape: (50, 54), z shape: (54, 1), b_f shape: (50, 1)
        temp_forget = np.dot(W_f, z) #dot product of weight of forget gate and concatenated input and hidden state
        temp_forget = temp_forget + b_f # apply bias
        forget = sigmoid(temp_forget) #apply activation function
        f_s.append(forget)
        #--------
        # Calculate input gate
        temp_input = np.dot(W_i, z) # dot product of weight of input gate and concatenated input and hidden state
        temp_input = temp_input + b_i # apply bias
        input = sigmoid(temp_input) # apply activation function
        i_s.append(input)
        #--------
        # Calculate candidate
        temp_candidate = np.dot(W_g, z) # dot product of weight of candidate and concatenated input and hidden state
        temp_candidate = temp_candidate + b_g # apply bias
        g_candidate = tanh(temp_candidate) # apply activation function
        g_s.append(g_candidate)
        #--------
        # Calculate memory state
        #   Now this is interesting because we say how much we should forget from the previous state (C_prev) and how much we should 
        #     add to the state (input * g_candidate), the forget or input values should range from 0 to 1, and we should expect something
        #     like this:  0.7 * C_prev + 0.99 * g_candidate, meaning: retain 70% of the previous state and add 99% of the candidate
        C_prev = forget * C_prev + input * g_candidate  # C_prev is passed as a parameter
        C_s.append(C_prev)
        #--------
        # Calculate output gate
        temp_output_gate = np.dot(W_o, z) # dot product of weight of output gate and concatenated input and hidden state
        temp_output_gate = temp_output_gate + b_o # apply bias
        output_gate = sigmoid(temp_output_gate) # apply activation function
        o_s.append(output_gate)
        #--------
        # Calculate hidden state
        hidden_state_prev = output_gate * tanh(C_prev)
        h_s.append(hidden_state_prev)
        #--------
        # Calculate logits
        #  "logits" refer to the raw, unnormalized scores that a model outputs before 
        #  applying an activation function like the softmax, as we can see below
        v_temp = np.dot(W_v, hidden_state_prev) # dot product of weights of the hidden state and the hidden state
        v = v_temp + b_v # apply bias
        v_s.append(v)
        #--------
        # Calculate softmax
        output_softmax = softmax(v)
        output_s.append(output_softmax)

    return z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, output_s
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part12(vocab_size, hidden_layer_size, params, idx_to_word, word_to_idx, test_set):
    print('PART 12')
    # Get first sentence in test set
    inputs, targets = test_set[1]

    # One-hot encode input and target sequence
    inputs_one_hot = one_hot_encode_sequence(sequence = inputs, vocab_size = vocab_size, word_to_idx=word_to_idx)
    targets_one_hot = one_hot_encode_sequence(sequence = targets, vocab_size = vocab_size, word_to_idx=word_to_idx)

    # Initialize hidden state as zeros
    h = np.zeros((hidden_layer_size, 1))
    c = np.zeros((hidden_layer_size, 1))

    #-----------
    # Forward pass
    z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward(
        inputs              = inputs_one_hot, 
        hidden_state_prev   = h, 
        C_prev              = c, 
        params              = params, 
        hidden_layer_size   = hidden_layer_size
    )
    #-----------

    output_sentence = [idx_to_word[np.argmax(output)] for output in outputs]
    print('Input sentence:')
    print(inputs)

    print('\nTarget sequence:')
    print(targets)

    print('\nPredicted sequence:')
    predicted = [idx_to_word[np.argmax(output)] for output in outputs]
    print(predicted)
    print('This is supposed to be junk - just ignore it')

    return{
        'inpus'             : inputs,
        'targets'           : targets,
        'inputs_one_hot'    : inputs_one_hot,
        'targets_one_hot'   : targets_one_hot,
        'h'                 : h,
        'c'                 : c,
        'z_s'               : z_s,
        'f_s'               : f_s,
        'i_s'               : i_s,
        'g_s'               : g_s,
        'C_s'               : C_s,
        'o_s'               : o_s,
        'h_s'               : h_s,
        'v_s'               : v_s,
        'outputs': outputs
    }
#-------------------------------------------------------------------------

part12_result = execute_part12(
    vocab_size          = part11_result['vocab_size'],
    hidden_layer_size   = part11_result['hidden_layer_size'],
    params              = part11_result['params'],
    idx_to_word         = part11_result['idx_to_word'],
    word_to_idx         = part11_result['word_to_idx'],
    test_set            = part11_result['test_set']
)

##########################################################################
##
##  PART 13
##
##########################################################################
#-------------------------------------------------------------------------
def clip_gradient_norm(grads, max_norm=0.25):
    {
    """
    Clips gradients to have a maximum norm of `max_norm`.
    This is to prevent the exploding gradients problem.
    """
    # rememnber: grads = d_U, d_V, d_W, d_b_hidden, d_b_out  
    #   U - weight input to hidden state
    #   V - weight matrix recurrent computation
    #   W - weight matrix hidden state to output
    #   bias_hidden shape
    #   bias_out
    # print(f'grads[0] U shape: {grads[0].shape}')
    # print(f'grads[1] V shape: {grads[1].shape}')
    # print(f'grads[2] W shape: {grads[2].shape}')
    # print(f'grads[3] b_hidden shape: {grads[3].shape}')
    # print(f'grads[4] b_out shape: {grads[4].shape}')
    }


    # Set the maximum of the norm to be of type float
    max_norm = float(max_norm)
    total_norm = 0

    
    # Calculate the L2 norm squared for each gradient and add them to the total norm
    for grad in grads:
        grad_norm = np.sum(np.power(grad, 2))
        total_norm += grad_norm
    
    total_norm = np.sqrt(total_norm)
    
    
    # Calculate clipping coeficient
    clip_coef = max_norm / (total_norm + 1e-6)

    # print(f'total_norm: {total_norm}') #from the example this should be around 28.2843
    # print(f'clip_coef: {clip_coef}') #from the example this would be around 0.008835
    
    #------------------
    # If the total norm is larger than the maximum allowable norm, then clip the gradient
    if clip_coef < 1:
        for grad in grads:
            # print(f'----')
            # print(f'grad: {grad}')
            # print(f'new grad: {grad * clip_coef}')
            grad *= clip_coef  # !!! KEEP THIS CODE!!!! grad = grad * clip_coef IS NOT STABLE. Python bug

    #------------------
    
    return grads
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def backward(z, f, i, g, C, o, h, v, hidden_layer_size, outputs, targets, params):
    """
    Arguments:
    z -- your concatenated input data  as a list of size m.
    f -- your forget gate computations as a list of size m.
    i -- your input gate computations as a list of size m.
    g -- your candidate computations as a list of size m.
    C -- your Cell states as a list of size m+1.
    o -- your output gate computations as a list of size m.
    h -- your Hidden state computations as a list of size m+1.
    v -- your logit computations as a list of size m.
    outputs -- your outputs as a list of size m.
    targets -- your targets as a list of size m.
    params -- python list containing:
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
    loss -- crossentropy loss for all elements in output
    grads -- lists of gradients of every element in p
    """

    # Unpack parameters
    W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = params 

    #-----------------------------------
    # Initialize gradients as zero
    W_f_d = np.zeros_like(W_f) # weights of the forget gate (derivative)
    b_f_d = np.zeros_like(b_f) # bias of the forget gate (derivative)

    W_i_d = np.zeros_like(W_i) # weights of the input gate (derivative)
    b_i_d = np.zeros_like(b_i) # bias of the input gate (derivative)

    W_g_d = np.zeros_like(W_g) # weights of the candidate (derivative)
    b_g_d = np.zeros_like(b_g) # bias of the candidate (derivative)

    W_o_d = np.zeros_like(W_o) # weights of the output gate (derivative)
    b_o_d = np.zeros_like(b_o) # bias of the output gate (derivative)

    W_v_d = np.zeros_like(W_v) # weights relating the hidden-state to the output (derivative)
    b_v_d = np.zeros_like(b_v) # bias relating the hidden-state to the output (derivative)
    
    # Set the next cell and hidden state equal to zero
    dh_next = np.zeros_like(h[0]) # hidden state (derivative)
    dC_next = np.zeros_like(C[0]) # cell state (derivative)
    #-----------------------------------
    # Track loss
    loss = 0

    print(f'outputs: {outputs}')
    print('----------')
    print(f'outputs[0]: {outputs[0]}')
        
    for t in reversed(range(len(outputs))): #outputs len: 14 (as our sentence has 14 words), so this will be a loop from 13 to 0
        

        # Compute cross-entropy loss
        {
        #  Remember we can have targets shape as (14, 4, 1) and outputs shape as (14, 4, 1), so what we do here
        #    is outputs[0]->(4,1) , targets[0]->(4,1)  
        #       
        # Formula: Loss += -(1/N) * SUM(i->n)[ y * log(y_hat + E) ] 
        #  note that 1/N*SUM(i->n) is the same as np.mean()
        }
        # Compute the cross entropy
        loss += -np.mean( np.log( outputs[t] + 1e-12 ) * targets[t] )
        # Get the previous hidden cell state
        C_prev= C[t-1]
        
        # Compute the derivative of the relation of the hidden-state to the output gate
        dv = np.copy(outputs[t])
        dv[np.argmax(targets[t])] -= 1

        # Update the gradient of the relation of the hidden-state to the output gate
        W_v_d += np.dot(dv, h[t].T)
        b_v_d += dv

        # Compute the derivative of the hidden state and output gate
        dh = np.dot(W_v.T, dv)        
        dh += dh_next
        do = dh * tanh(C[t])
        do = sigmoid(o[t], derivative=True)*do
        
        # Update the gradients with respect to the output gate
        W_o_d += np.dot(do, z[t].T)
        b_o_d += do

        # Compute the derivative of the cell state and candidate g
        dC = np.copy(dC_next)
        dC += dh * o[t] * tanh(tanh(C[t]), derivative=True)
        dg = dC * i[t]
        dg = tanh(g[t], derivative=True) * dg
        
        # Update the gradients with respect to the candidate
        W_g_d += np.dot(dg, z[t].T)
        b_g_d += dg

        # Compute the derivative of the input gate and update its gradients
        di = dC * g[t]
        di = sigmoid(i[t], True) * di
        W_i_d += np.dot(di, z[t].T)
        b_i_d += di

        # Compute the derivative of the forget gate and update its gradients
        df = dC * C_prev
        df = sigmoid(f[t]) * df
        W_f_d += np.dot(df, z[t].T)
        b_f_d += df

        # Compute the derivative of the input and update the gradients of the previous hidden and cell state
        dz = (np.dot(W_f.T, df)
             + np.dot(W_i.T, di)
             + np.dot(W_g.T, dg)
             + np.dot(W_o.T, do))
        dh_prev = dz[:hidden_layer_size, :]
        dC_prev = f[t] * dC
        
    grads= W_f_d, W_i_d, W_g_d, W_o_d, W_v_d, b_f_d, b_i_d, b_g_d, b_o_d, b_v_d
    
    # Clip gradients
    grads = clip_gradient_norm(grads)
    
    return loss, grads
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part13(z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, hidden_layer_size, outputs, targets_one_hot, params):
    print('PART 13')

    
    
    # Perform a backward pass
    loss, grads = backward(z = z_s, f = f_s, i = i_s, g = g_s, C = C_s, o = o_s, h = h_s, v = v_s, 
                           hidden_layer_size= hidden_layer_size, outputs = outputs, targets=targets_one_hot, params = params)

    print('We get a loss of:')
    print(loss)

    return {
        'loss': loss,
        'grads': grads
    }

#-------------------------------------------------------------------------
execute_part13(
    z_s                 = part12_result['z_s'],
    f_s                 = part12_result['f_s'],
    i_s                 = part12_result['i_s'],
    g_s                 = part12_result['g_s'],
    C_s                 = part12_result['C_s'],
    o_s                 = part12_result['o_s'],
    h_s                 = part12_result['h_s'],
    v_s                 = part12_result['v_s'],
    hidden_layer_size   = part11_result['hidden_layer_size'],
    outputs             = part12_result['outputs'],
    targets_one_hot     = part12_result['targets_one_hot'],
    params              = part11_result['params']
)

exit()
##########################################################################
##
##  PART 14
##
##########################################################################
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------
def update_parameters(params, grads, learning_rate=1e-3):
    # To train our network, we need an optimizer. A common method is gradient descent,
    # which updates parameters using the rule: θ_{n+1} = (θ_{n}) - (η * ∂E/∂θ_{n}),
    #     θ = Theta - often used to represent parameters or weights of a model
    #     η = Eta - often used to represent the learning rate
    #     E = Cost function
    #     ∂E/∂θ = Partial derivative of the cost function with respect to the parameter
    #     η * ∂E/∂θ_{n} = step size
    #     
    # Quick note: Gradients (grads) - These are the partial derivatives of the cost function ( E ) 
    #   with respect to each parameter. They indicate how much the cost function would change if 
    #   the parameter is adjusted.


    # This whole function is similar to what happens when you run `optimizer.step()` in PyTorch with SGD.

    # Take a step
    for param, grad in zip(params, grads):
        param -= learning_rate * grad # Keep this code, anything else is not stable

    return params
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def train_LSTM(hidden_layer_size, vocab_size, word_to_idx,  training_set, validation_set):
    # num_epochs = 85
    num_epochs = 50

    # Initialize a new network
    z_size = hidden_layer_size + vocab_size # Size of concatenated hidden + input vector
    params = init_lstm(hidden_layer_size = hidden_layer_size, vocab_size = vocab_size, z_size = z_size)

    # Initialize hidden state as zeros
    hidden_state = np.zeros((hidden_layer_size, 1))

    # Track loss
    training_loss, validation_loss = [], []

    # For each epoch
    for i in range(num_epochs):
        
        # Track loss
        epoch_training_loss = 0
        epoch_validation_loss = 0
        
        # For each sentence in validation set
        for inputs, targets in validation_set:
            
            # One-hot encode input and target sequence
            inputs_one_hot = one_hot_encode_sequence(sequence = inputs, vocab_size = vocab_size, word_to_idx = word_to_idx)
            targets_one_hot = one_hot_encode_sequence(sequence = targets, vocab_size = vocab_size, word_to_idx = word_to_idx)

            # Initialize hidden state and cell state as zeros
            h = np.zeros((hidden_layer_size, 1))
            c = np.zeros((hidden_layer_size, 1))

            # Forward pass
            z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward(inputs = inputs_one_hot, hidden_state_prev  = h, C_prev = c, params = params, hidden_layer_size=hidden_layer_size)
            
            # Backward pass
            loss, _ = backward(
                z = z_s, 
                f = f_s, 
                i = i_s, 
                g = g_s, 
                C = C_s, 
                o = o_s, 
                h = h_s, 
                v = v_s,
                hidden_layer_size   = hidden_layer_size, 
                outputs             = outputs, 
                targets             = targets_one_hot, 
                params              = params
            )
            
            # Update loss
            epoch_validation_loss += loss
        
        # For each sentence in training set
        for inputs, targets in training_set:
            
            # One-hot encode input and target sequence
            inputs_one_hot = one_hot_encode_sequence(sequence = inputs, vocab_size = vocab_size,word_to_idx = word_to_idx)
            targets_one_hot = one_hot_encode_sequence(sequence = targets, vocab_size = vocab_size, word_to_idx = word_to_idx)

            # Initialize hidden state and cell state as zeros
            h = np.zeros((hidden_layer_size, 1))
            c = np.zeros((hidden_layer_size, 1))

            # Forward pass
            z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward(inputs = inputs_one_hot, hidden_state_prev=h, C_prev=c, params=params, hidden_layer_size=hidden_layer_size)
            
            # Backward pass
            loss, grads = backward(
                z = z_s, 
                f = f_s, 
                i = i_s, 
                g = g_s, 
                C = C_s, 
                o = o_s, 
                h = h_s, 
                v = v_s, 
                hidden_layer_size   = hidden_layer_size, 
                outputs             = outputs, 
                targets             = targets_one_hot, 
                params              = params
            )
            
            # Update parameters
            params = update_parameters(params = params, grads = grads, learning_rate=1e-1)
            
            # Update loss
            epoch_training_loss += loss
                    
        # Save loss for plot
        training_loss.append(epoch_training_loss/len(training_set))
        validation_loss.append(epoch_validation_loss/len(validation_set))

        # Print loss every 5 epochs
        if i % 5 == 0:
            print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')

    return {
        'params' : params,
        'training_loss': training_loss,
        'validation_loss': validation_loss
    }

def make_prediction(hidden_layer_size, vocab_size, idx_to_word, word_to_idx, params, test_set):
        
    # Get first sentence in test set
    inputs, targets = test_set[1]

    # One-hot encode input and target sequence
    inputs_one_hot = one_hot_encode_sequence(sequence=inputs, vocab_size=vocab_size, word_to_idx=word_to_idx)
    targets_one_hot = one_hot_encode_sequence(sequence=targets, vocab_size=vocab_size, word_to_idx=word_to_idx)

    # Initialize hidden state as zeros
    h = np.zeros((hidden_layer_size, 1))
    c = np.zeros((hidden_layer_size, 1))

    # Forward pass
    z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward(inputs=inputs_one_hot, hidden_state_prev=h, C_prev=c, params=params, hidden_layer_size=hidden_layer_size)

    # Print example
    print('Input sentence:')
    print(inputs)

    print('\nTarget sequence:')
    print(targets)

    print('\nPredicted sequence:')
    predicted = [idx_to_word[np.argmax(output)] for output in outputs]
    print(predicted)

    print(f'Is predicted equal to target?: {predicted == targets}')  

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
def execute_part14(hidden_layer_size, vocab_size, word_to_idx, idx_to_word, training_set, validation_set, test_set):
    print('PART 14')
    train_LSTM_result = train_LSTM(
        hidden_layer_size   = hidden_layer_size,
        vocab_size          = vocab_size,
        word_to_idx         = word_to_idx,
        training_set        = training_set,
        validation_set      = validation_set,
    )
    make_prediction(
        hidden_layer_size   = hidden_layer_size,
        vocab_size          = vocab_size,
        idx_to_word         = idx_to_word,
        word_to_idx         = word_to_idx,
        params              = train_LSTM_result['params'],
        test_set            = test_set
    )
    plot_graph(
        training_loss       = train_LSTM_result['training_loss'],
        validation_loss     = train_LSTM_result['validation_loss']
    )
#-------------------------------------------------------------------------
execute_part14(
    hidden_layer_size   = part11_result['hidden_layer_size'],
    vocab_size          = part11_result['vocab_size'],
    word_to_idx         = part11_result['word_to_idx'],
    idx_to_word         = part11_result['idx_to_word'],
    training_set        = part11_result['training_set'],
    validation_set      = part11_result['validation_set'],
    test_set            = part11_result['test_set']
)