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
        # YOUR CODE HERE!
        word_to_idx[word] = idx
        idx_to_word[idx] = word

    return word_to_idx, idx_to_word, num_sentences, vocab_size
#-------------------------------------------------------------------------

word_to_idx, idx_to_word, num_sequences, vocab_size = sequences_to_dicts(sequences)

print(f'We have {num_sequences} sentences and {len(word_to_idx)} unique tokens in our dataset (including UNK).\n')
print('The index of \'b\' is', word_to_idx['b'])
print(f'The word corresponding to index 1 is \'{idx_to_word[1]}\'')
print('------')
print(f'vocab_size: {vocab_size}')
for i in range(vocab_size):
    print(f'{i}: {idx_to_word[i]}')

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

training_set, validation_set, test_set = create_datasets(sequences = sequences, dataset_class = Dataset)

print(f'We have {len(training_set)} samples in the training set.')
print(f'We have {len(validation_set)} samples in the validation set.')
print(f'We have {len(test_set)} samples in the test set.')


word_to_idx, idx_to_word, num_sequences, vocab_size = sequences_to_dicts(sequences = sequences)
print('------')
print(f'We have {num_sequences} sentences and {len(word_to_idx)} unique tokens in our dataset (including UNK).\n')
print('The index of \'b\' is', word_to_idx['b'])
print(f'The word corresponding to index 1 is \'{idx_to_word[1]}\'')
print('------')
print(f'test_set length: {len(test_set)}')
print(f'test_set[0]: {test_set[0]} - Considering the input was [a, b] we would expect the target to be [b, EOS]')



##########################################################################
##
##  PART 4
##
##########################################################################

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
def one_hot_encode_sequence(sequence, vocab_size, param_word_to_idx):
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
print(f'The one-hot encoding of \'a\' is:\n {test_word}.')

test_sentence = one_hot_encode_sequence(sequence = ['a', 'b'], vocab_size = vocab_size, param_word_to_idx = word_to_idx)
print(f'Our one-hot encoding of \'a b\' has shape {test_sentence.shape}.')
print(f'The one-hot encoding of \'a b\' is:')
print('a:\n----')
print(test_sentence[0])
print('b:\n----')
print(test_sentence[1])
print('----')

##########################################################################
##
##  PART 5
##
##########################################################################

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
def init_rnn(hidden_size, vocab_size):
    """
    Initializes our recurrent neural network.
    
    Args:
     `hidden_size`: the dimensions of the hidden state
     `vocab_size`: the dimensions of our vocabulary
    """

    #---------------------------------------
    # Part 1
    # Initialize the weights with zeros
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
    #---------------------------------------
    # Part 2
    # Orthogonal initialization of the weight matrices
    U = init_orthogonal(U)
    V = init_orthogonal(V)
    W = init_orthogonal(W)
    #---------------------------------------
    
    # Return parameters as a tuple
    return U, V, W, b_hidden, b_out
#-------------------------------------------------------------------------

hidden_layer_size = 50 # Number of hidden units in this layer
vocab_size  = len(word_to_idx) # Size of the vocabulary used

params = init_rnn(hidden_size=hidden_layer_size, vocab_size=vocab_size)
print('----------------------------------------------')
print(f'U (weight input to hidden state) shape: {params[0].shape}') # U
print(f'V (weight matrix recurrent computation) shape: {params[1].shape}') # V
print(f'W (weight matrix hidden state to output) shape: {params[2].shape}') # W
print(f'bias_hidden shape: {params[3].shape}') # b_hidden
print(f'bias_out shape: {params[4].shape}') # b_out
print('----------------------------------------------')
print(f'U first 5 rows:\n------------\n{params[0][:5]}\n------------\n')
print(f'V first 2 rows:\n------------\n{params[1][:2]}\n------------\n')
print(f'W first 5 rows:\n------------\n{params[2][:2]}\n------------\n')
print(f'bias_hidden first 5 rows:{params[3]}')
print(f'bias_out :{params[4]}')

##########################################################################
##
##  PART 6
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


##########################################################################
##
##  PART 7
##
##########################################################################
#-------------------------------------------------------------------------
def forward_pass(inputs, hidden_state, params_U_V_W_bhidden_bout):
    """
    Computes the forward pass of a vanilla RNN.
    Args:
     `inputs`: sequence of inputs to be processed
     `hidden_state`: an already initialized hidden state
     `params`: the parameters of the RNN (U, V, W, b_hidden, b_out)
    """
    #---------
    # First we unpack our parameters
    #   U - weight input to hidden state, 
    #   V - weight matrix recurrent computation,
    #   W - weight matrix hidden state to output, 
    #   bias_hidden shape, 
    #   bias_out
    U, V, W, b_hidden, b_out = params_U_V_W_bhidden_bout
    # print(f'  U shape: {U.shape}')
    # print(f'  V shape: {V.shape}')
    # print(f'  W shape: {W.shape}')
    # print(f'  b_hidden shape: {b_hidden.shape}')
    # print(f'  b_out shape: {b_out.shape}')
    
    # Create a list to store outputs and hidden states
    outputs, hidden_states = [], []
    #---------
    # For each element in input sequence
    for t in range(len(inputs)): # t as the notation for time-step
        # Compute new hidden state
       
        dot_U_inputsT = np.dot(U, inputs[t]) # U * inputs(t)
        dot_V_hidden_state = np.dot(V, hidden_state)
        temp_hidden_state = dot_U_inputsT + dot_V_hidden_state + b_hidden

        # print('---')
        # print(f'  inputs[t] shape: {inputs[t].shape}')
        # print(f'  dot_U_inputsT shape: {dot_U_inputsT.shape}')
        # print(f'  dot_V_hidden_state shape: {dot_V_hidden_state.shape}')
        # print(f'  temp_hidden_state shape: {temp_hidden_state.shape}')
        #---
        hidden_state = tanh(temp_hidden_state)

        # Compute output
        temp_out = np.dot(W, hidden_state) + b_out
        out = softmax(temp_out)
        
        # Save results and continue
        outputs.append(out)
        hidden_states.append(hidden_state.copy())
    #---------
    return outputs, hidden_states
#-------------------------------------------------------------------------
print('----------------------------------------------')
print('about the inputs and targets remember:')
print('  example: \'The quick brown fox jumps\'')
print('      Inputs:  ["The", "quick", "brown", "fox"]')
print('      Targets: ["quick", "brown", "fox", "jumps"]')
print(f'training_set len: {len(training_set)}')
print(f'training_set[0][0] (inputs)\ntraining_set[0][1] (targets):\n{training_set[0][0]}\n{training_set[0][1]}')
print('----------------------------------------------')

#-------------------
# Get first sequence in training set
test_input_sequence, test_target_sequence = training_set[0]

# One-hot encode input and target sequence
test_input = one_hot_encode_sequence(
        sequence = test_input_sequence, vocab_size = vocab_size, param_word_to_idx = word_to_idx
    )
test_target = one_hot_encode_sequence(
        sequence = test_target_sequence, vocab_size = vocab_size, param_word_to_idx = word_to_idx
    )

# Initialize hidden state as zeros
global_hidden_state = np.zeros((hidden_layer_size, 1)) # hidden_layer_size = 50

# Now let's try out our new function
global_outputs, global_hidden_states = forward_pass(
        inputs = test_input, hidden_state = global_hidden_state, params_U_V_W_bhidden_bout = params
    )

#-------------------
print('Input sequence:')
print(test_input_sequence)

print('\nTarget sequence:')
print(test_target_sequence)

print('\nPredicted sequence:')
print([idx_to_word[np.argmax(output)] for output in global_outputs])
print('Note: At this stage the predictions are random, as the model has not been trained yet.')
print('----------------------------------------------')
#-------------------
##########################################################################
##
##  PART 8
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
            grad = grad * clip_coef
    #------------------
    
    return grads
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def backward_pass(inputs, outputs, hidden_states, targets, params_U_V_W_bhidden_bout):
    {"""
    Computes the backward pass of a vanilla RNN.
    Args:
     `inputs`: sequence of inputs to be processed
     `outputs`: sequence of outputs from the forward pass
     `hidden_states`: sequence of hidden_states from the forward pass
     `targets`: sequence of targets
     `params`: the parameters of the RNN (U, V, W, b_hidden, b_out)
    """
    }
    #-------------------
    # First we unpack our parameters
    U, V, W, b_hidden, b_out = params_U_V_W_bhidden_bout

    {# print(f'U shape: {U.shape}')
    # print(f'V shape: {V.shape}')
    # print(f'W shape: {W.shape}')
    # print(f'b_hidden shape: {b_hidden.shape}')
    # print(f'b_out shape: {b_out.shape}')
    }
    #-------------------
    # Initialize gradients as zero
    d_U, d_V, d_W = np.zeros_like(U), np.zeros_like(V), np.zeros_like(W) # np.zeros_like(var) generates a new array of zeros with the same shape and type as the array 'var'
    d_b_hidden, d_b_out = np.zeros_like(b_hidden), np.zeros_like(b_out)
    
    # Keep track of hidden state derivative and loss
    d_h_next = np.zeros_like(hidden_states[0])
    loss = 0
    {
    #-------------------
    # For each element in output sequence
    # NB: We iterate backwards s.t. t = N, N-1, ... 1, 0
    #-------------------
    # len_outputs = len(outputs)
    # print(f'len(outputs): {len_outputs}')
    # print(f'range(len(outputs)): {range(len_outputs)}')
    # print(f'reversed( range( len(outputs) ) ): {reversed( range( len(outputs) ) )}')
    }
    for t in reversed( range( len(outputs) ) ): #for a sequence of 14 elements, this will be 13, 12, 11, ..., 0
        # Compute cross-entropy loss (as a scalar)
        {
        #  Remember we can have targets shape as (14, 4, 1) and outputs shape as (14, 4, 1), so what we do here
        #    is outputs[0]->(4,1) , targets[0]->(4,1)  
        #       
        # Formula: Loss += -(1/N) * SUM(i->n)[ y * log(y_hat + E) ] 
        #  note that 1/N*SUM(i->n) is the same as np.mean()
        }
        loss += -np.mean( np.log( outputs[t]+1e-12 ) * targets[t] )
        
        {
        # print(f'outputs[t]: {outputs[t]}')
        # print(f'targets[t]: {targets[t]}')
        # print(f'np.log( outputs[t]): {np.log( outputs[t] )}')
        # print(f'np.log( outputs[t]+1e-12 ) * targets[t]: { np.log( outputs[t]+1e-12 ) * targets[t] }')
        # print(f'np.mean( np.log( outputs[t]+1e-12 ) * targets[t] ): {np.mean( np.log( outputs[t]+1e-12 ) * targets[t] )}')
        # print('--------')
        #-------------------
        }
        
        # Backpropagate into output (derivative of cross-entropy)
        {
        # if you're confused about this step, see this link for an explanation:
        # http://cs231n.github.io/neural-networks-case-study/#grad
        #   Suppose outputs[t] is [0.1, 0.7, 0.2] and targets[t] is [0, 1, 0] (one-hot encoded)
        # The code would execute as follows:
        #     d_o = outputs[t].copy() results in d_o = [0.1, 0.7, 0.2].
        #     np.argmax(targets[t]) returns 1 (the index of the maximum value in targets[t]).
        #     d_o[1] -= 1 modifies d_o to [0.1, -0.3, 0.2].
        }
        d_o = outputs[t].copy()
        d_o[ np.argmax(targets[t]) ] -= 1
        
        # Backpropagate into W (W - weight matrix hidden state to output)
        d_W += np.dot(d_o, hidden_states[t].T)
        d_b_out += d_o
        
        # Backpropagate into h (hidden state)
        d_h = np.dot(W.T, d_o) + d_h_next
        
        # Backpropagate through non-linearity
        d_f = tanh(hidden_states[t], derivative=True) * d_h
        d_b_hidden += d_f
        
        # Backpropagate into U (U - weight input to hidden state)
        d_U += np.dot(d_f, inputs[t].T)
        
        # Backpropagate into V (V - weight matrix recurrent computation)
        d_V += np.dot(d_f, hidden_states[t-1].T)
        d_h_next = np.dot(V.T, d_f)
    #-------------------
    # Pack gradients
    grads = d_U, d_V, d_W, d_b_hidden, d_b_out    
    
    # Clip gradients
    grads = clip_gradient_norm(grads)
    
    return loss, grads
#-------------------------------------------------------------------------

print(f'To clarify, the original sentence has a length of 14 elements')
print(f'test_input shape: {test_input.shape}') # (14, 4, 1)
# print(f'test_input: {test_input}') # 'a' -> [[1],[0],[0],[0]], 'a' -> [[1],[0],[0],[0]], ...
print(f'global_outputs len: {len(global_outputs)}') 
# print(f'global_outputs: {global_outputs}') # at this stage this is mostly random junk
print(f'global_hidden_states len: {len(global_hidden_states)}')
print(f'global_hidden_states shape: {global_hidden_states[0].shape}') # 14*(50, 1)
# print(f'global_hidden_states: {global_hidden_states}') # at this stage this is mostly random junk

print('Remember: While the test input is valid, at this stage the global outputs and hidden states are random junk')
loss, grads = backward_pass(
        inputs = test_input, outputs = global_outputs, hidden_states = global_hidden_states, 
        targets = test_target, params_U_V_W_bhidden_bout = params
    )

print('We get a loss of:')
print(loss)

##########################################################################
##
##  PART 9
##
##########################################################################
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
        temp_param = param - learning_rate * grad
        param = temp_param
    
    return params
#-------------------------------------------------------------------------

import matplotlib.pyplot as plt
# %matplotlib inline
# #-------------------------------------------------------------------------
# def train_rnn(param_hidden_layer_size, param_vocab_size, num_epochs = 1000):
#     #------------
#     # Initialize a new network
#     params = init_rnn(hidden_size = param_hidden_layer_size, vocab_size = param_vocab_size)

#     # Initialize hidden state as zeros
#     local_hidden_state = np.zeros((param_hidden_layer_size, 1))

#     # Track loss
#     training_loss, validation_loss = [], []
#     #------------

#     #-------------------------------------------------------------------------
#     # For each epoch
#     for i in range(num_epochs):
        
#         # Track loss
#         epoch_training_loss = 0
#         epoch_validation_loss = 0
        
#         # For each sentence in validation set
#         for inputs, targets in validation_set:
#             # One-hot encode input and target sequence
#             inputs_one_hot = one_hot_encode_sequence(sequence = inputs, vocab_size = vocab_size, param_word_to_idx = word_to_idx)
#             targets_one_hot = one_hot_encode_sequence(sequence = targets, vocab_size = vocab_size, param_word_to_idx = word_to_idx)
            
#             # Re-initialize hidden state
#             global_hidden_state = np.zeros_like(global_hidden_state)

#             # Forward pass
#             global_outputs, global_hidden_states = forward_pass(
#                 inputs = inputs_one_hot, hidden_state = global_hidden_state, params_U_V_W_bhidden_bout = params
#             )

#             # Backward pass - returns loss and grads ( _ )
#             loss, _ = backward_pass(inputs = inputs_one_hot, outputs = global_outputs, 
#                 hidden_states = global_hidden_states, targets = targets_one_hot, params_U_V_W_bhidden_bout = params
#             )
            
#             # Update loss
#             epoch_validation_loss += loss
        
#         # For each sentence in training set
#         for inputs, targets in training_set:
#             # One-hot encode input and target sequence
#             inputs_one_hot = one_hot_encode_sequence(sequence = inputs, vocab_size = vocab_size, param_word_to_idx = word_to_idx)
#             targets_one_hot = one_hot_encode_sequence(sequence = targets, vocab_size = vocab_size, param_word_to_idx = word_to_idx)
            
#             # Re-initialize hidden state
#             global_hidden_state = np.zeros_like(global_hidden_state)

#             # Forward pass
#             global_outputs, global_hidden_states = forward_pass(
#                 inputs = inputs_one_hot, hidden_state = global_hidden_state, params_U_V_W_bhidden_bout = params
#             )

#             # Backward pass
#             loss, grads = backward_pass(
#                 inputs = inputs_one_hot, outputs = global_outputs, hidden_states = global_hidden_states, 
#                 targets = targets_one_hot, params_U_V_W_bhidden_bout = params
#             )
            
#             if np.isnan(loss): #not a number
#                 raise ValueError('Gradients have vanished!')
            
#             # Update parameters
#             params = update_parameters(params, grads, learning_rate=3e-4)
            
#             # Update loss
#             epoch_training_loss += loss
            
#         # Save loss for plot
#         training_loss.append(epoch_training_loss/len(training_set))
#         validation_loss.append(epoch_validation_loss/len(validation_set))

#         # Print loss every 100 epochs
#         if i % 100 == 0:
#             print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')
#     #-------------------------------------------------------------------------
# #-------------------------------------------------------------------------

# Hyper-parameters
num_epochs = 1000

# Initialize a new network
params = init_rnn(hidden_size=hidden_layer_size, vocab_size=vocab_size)

# Initialize hidden state as zeros
global_hidden_state = np.zeros((hidden_layer_size, 1))

# Track loss
training_loss, validation_loss = [], []

#-------------------------------------------------------------------------
# For each epoch
for i in range(num_epochs):
    
    # Track loss
    epoch_training_loss = 0
    epoch_validation_loss = 0
    
     # For each sentence in validation set
    for inputs, targets in validation_set:
        # One-hot encode input and target sequence
        inputs_one_hot = one_hot_encode_sequence(sequence = inputs, vocab_size = vocab_size, param_word_to_idx = word_to_idx)
        targets_one_hot = one_hot_encode_sequence(sequence = targets, vocab_size = vocab_size, param_word_to_idx = word_to_idx)
        
        # Re-initialize hidden state
        global_hidden_state = np.zeros_like(global_hidden_state)

        # Forward pass
        global_outputs, global_hidden_states = forward_pass(
            inputs = inputs_one_hot, hidden_state = global_hidden_state, params_U_V_W_bhidden_bout = params
        )

        # Backward pass - returns loss and grads ( _ )
        loss, _ = backward_pass(inputs = inputs_one_hot, outputs = global_outputs, 
            hidden_states = global_hidden_states, targets = targets_one_hot, params_U_V_W_bhidden_bout = params
        )
        
        # Update loss
        epoch_validation_loss += loss
    
    # For each sentence in training set
    for inputs, targets in training_set:
        # One-hot encode input and target sequence
        inputs_one_hot = one_hot_encode_sequence(sequence = inputs, vocab_size = vocab_size, param_word_to_idx = word_to_idx)
        targets_one_hot = one_hot_encode_sequence(sequence = targets, vocab_size = vocab_size, param_word_to_idx = word_to_idx)
        
        # Re-initialize hidden state
        global_hidden_state = np.zeros_like(global_hidden_state)

        # Forward pass
        global_outputs, global_hidden_states = forward_pass(
            inputs = inputs_one_hot, hidden_state = global_hidden_state, params_U_V_W_bhidden_bout = params
        )

        # Backward pass
        loss, grads = backward_pass(
            inputs = inputs_one_hot, outputs = global_outputs, hidden_states = global_hidden_states, 
            targets = targets_one_hot, params_U_V_W_bhidden_bout = params
        )
        
        if np.isnan(loss): #not a number
            raise ValueError('Gradients have vanished!')
        
        # Update parameters
        params = update_parameters(params, grads, learning_rate=3e-4)
        
        # Update loss
        epoch_training_loss += loss
        
    # Save loss for plot
    training_loss.append(epoch_training_loss/len(training_set))
    validation_loss.append(epoch_validation_loss/len(validation_set))

    # Print loss every 100 epochs
    if i % 100 == 0:
        print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')
#-------------------------------------------------------------------------

# Get first sentence in test set
inputs, targets = test_set[1]

# One-hot encode input and target sequence
inputs_one_hot = one_hot_encode_sequence(sequence = inputs, vocab_size = vocab_size, param_word_to_idx = word_to_idx)
targets_one_hot = one_hot_encode_sequence(sequence = targets, vocab_size = vocab_size, param_word_to_idx = word_to_idx)

# Initialize hidden state as zeros
global_hidden_state = np.zeros((hidden_layer_size, 1))

# Forward pass
global_outputs, global_hidden_states = forward_pass(inputs_one_hot, global_hidden_state, params)
output_sentence = [idx_to_word[np.argmax(output)] for output in global_outputs]
print('Input sentence:')
print(inputs)

print('\nTarget sequence:')
print(targets)


print('\nPredicted sequence:')
translated_output = [idx_to_word[np.argmax(output)] for output in global_outputs]
print(translated_output)


print(f'Is the target sequence equal to the predicted sequence? {targets == translated_output}')

# Plot training and validation loss
epoch = np.arange(len(training_loss))
plt.figure()
plt.plot(epoch, training_loss, 'r', label='Training loss',)
plt.plot(epoch, validation_loss, 'b', label='Validation loss')
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('NLL')
plt.show()
exit()
##########################################################################
##
##  PART 10
##
##########################################################################

#-------------------------------------------------------------------------
def freestyle(params, sentence = '', num_generate = 4 , param_hidden_size = 50):
    """
    Takes in a sentence as a string and outputs a sequence
    based on the predictions of the RNN.
    Args:
     `params`: the parameters of the network
     `sentence`: string with whitespace-separated tokens
     `num_generate`: the number of tokens to generate
    """

    sentence = sentence.split(' ')
    
    sentence_one_hot = one_hot_encode_sequence(sentence = sentence, vocab_size = vocab_size, param_word_to_idx = word_to_idx)
    
    # Initialize hidden state as zeros
    hidden_state = np.zeros((param_hidden_size, 1))

    # Generate hidden state for sentence
    outputs, hidden_states = forward_pass(inputs= sentence_one_hot, hidden_state =  hidden_state, params_U_V_W_bhidden_bout = params)
    
    # Output sentence
    output_sentence = sentence
    
    # Append first prediction
    word = idx_to_word[np.argmax(outputs[-1])]  #return the index of the max value in the array 
    output_sentence.append(word)
    
    # Forward pass
    for i in range(num_generate):

        # Get the latest prediction and latest hidden state
        output = outputs[-1]
        hidden_state = hidden_states[-1]
    
        # Reshape our output to match the input shape of our forward pass
        output = output.reshape(1, output.shape[0], output.shape[1])
    
        # Forward pass
        outputs, hidden_states = forward_pass(
            inputs = output, hidden_state = hidden_state, params_U_V_W_bhidden_bout = params
        )
        
        # Compute the index the most likely word and look up the corresponding word
        word = idx_to_word[np.argmax(outputs)]
        
        output_sentence.append(word)
        
    return output_sentence
#------------------------------------------------------------------------- 

# Perform freestyle
result_freestyle = freestyle(params = params, sentence='a a a a a b')
print(f'Result freestyle\n:{result_freestyle}')
