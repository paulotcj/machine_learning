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


##########################################################################
##
##  PART 7
##
##########################################################################
#-------------------------------------------------------------------------
def forward_pass(inputs, hidden_state, params):
    # Computes the forward pass of a vanilla RNN.
    # Args:
    #  `inputs`: sequence of inputs to be processed
    #  `hidden_state`: an already initialized hidden state
    #  `params`: the parameters of the RNN

    # First we unpack our parameters
    #   U - weight input to hidden state, V - weight matrix recurrent computation,
    #   W - weight matrix hidden state to output, bias_hidden shape, bias_out
    U, V, W, b_hidden, b_out = params
    
    # Create a list to store outputs and hidden states
    outputs, hidden_states = [], []
    
    # For each element in input sequence
    for t in range(len(inputs)): # t as the notation for time-step
        # Compute new hidden state
        temp_hidden_state = np.dot(U, inputs[t]) + np.dot(V, hidden_state) + b_hidden
        hidden_state = tanh(temp_hidden_state)

        # Compute output
        temp_out = np.dot(W, hidden_state) + b_out
        out = softmax(temp_out)
        
        # Save results and continue
        outputs.append(out)
        hidden_states.append(hidden_state.copy())
    
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

# Get first sequence in training set
test_input_sequence, test_target_sequence = training_set[0]

# One-hot encode input and target sequence
test_input = one_hot_encode_sequence(sequence = test_input_sequence, vocab_size = vocab_size, param_word_to_idx = word_to_idx)
test_target = one_hot_encode_sequence(sequence = test_target_sequence, vocab_size = vocab_size, param_word_to_idx = word_to_idx)

# Initialize hidden state as zeros
hidden_state = np.zeros((hidden_size, 1))

# Now let's try out our new function
outputs, hidden_states = forward_pass(test_input, hidden_state, params)

print('Input sequence:')
print(test_input_sequence)

print('\nTarget sequence:')
print(test_target_sequence)

print('\nPredicted sequence:')
print([idx_to_word[np.argmax(output)] for output in outputs])
print('Note: At this stage the predictions are random, as the model has not been trained yet.')
print('----------------------------------------------')


##########################################################################
##
##  PART 8
##
##########################################################################

#-------------------------------------------------------------------------
def clip_gradient_norm(grads, max_norm=0.25):
    # Clips gradients to have a maximum norm of `max_norm`.
    # This is to prevent the exploding gradients problem.

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
    
    # If the total norm is larger than the maximum allowable norm, then clip the gradient
    if clip_coef < 1:
        for grad in grads:
            grad *= clip_coef
    
    return grads
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def backward_pass(inputs, outputs, hidden_states, targets, params):
    # Computes the backward pass of a vanilla RNN.
    # Args:
    #  `inputs`: sequence of inputs to be processed
    #  `outputs`: sequence of outputs from the forward pass
    #  `hidden_states`: sequence of hidden_states from the forward pass
    #  `targets`: sequence of targets
    #  `params`: the parameters of the RNN

    # First we unpack our parameters
    U, V, W, b_hidden, b_out = params
    
    # Initialize gradients as zero
    d_U, d_V, d_W = np.zeros_like(U), np.zeros_like(V), np.zeros_like(W) # np.zeros_like(var) generates a new array of zeros with the same shape and type as the array 'var'
    d_b_hidden, d_b_out = np.zeros_like(b_hidden), np.zeros_like(b_out)
    
    # Keep track of hidden state derivative and loss
    d_h_next = np.zeros_like(hidden_states[0])
    loss = 0
    
    # For each element in output sequence
    # NB: We iterate backwards s.t. t = N, N-1, ... 1, 0
    #----
    for t in reversed( range( len(outputs) ) ):
        #------------------
        # Compute cross-entropy loss (as a scalar)
        #  Remember we can have targets shape as (14, 4, 1) and outputs shape as (14, 4, 1), so what we do here
        #    is outputs[0]->(4,1) , targets[0]->(4,1)  
        #       
        # Formula: Loss += -(1/N) * SUM(i->n)[ y * log(y_hat + E) ] 
        #  note that 1/N*SUM(i->n) is the same as np.mean()
        loss += -np.mean( np.log( outputs[t]+1e-12 ) * targets[t] )
        
        # print(f'outputs[t]: {outputs[t]}')
        # print(f'targets[t]: {targets[t]}')
        # print(f'np.log( outputs[t]): {np.log( outputs[t] )}')
        # print(f'np.log( outputs[t]+1e-12 ) * targets[t]: { np.log( outputs[t]+1e-12 ) * targets[t] }')
        # print(f'np.mean( np.log( outputs[t]+1e-12 ) * targets[t] ): {np.mean( np.log( outputs[t]+1e-12 ) * targets[t] )}')
        # print('--------')
        #------------------
        
        # Backpropagate into output (derivative of cross-entropy)
        # if you're confused about this step, see this link for an explanation:
        # http://cs231n.github.io/neural-networks-case-study/#grad
        d_o = outputs[t].copy()
        d_o[ np.argmax(targets[t]) ] -= 1
        
        # Backpropagate into W
        d_W += np.dot(d_o, hidden_states[t].T)
        d_b_out += d_o
        
        # Backpropagate into h
        d_h = np.dot(W.T, d_o) + d_h_next
        
        # Backpropagate through non-linearity
        d_f = tanh(hidden_states[t], derivative=True) * d_h
        d_b_hidden += d_f
        
        # Backpropagate into U
        d_U += np.dot(d_f, inputs[t].T)
        
        # Backpropagate into V
        d_V += np.dot(d_f, hidden_states[t-1].T)
        d_h_next = np.dot(V.T, d_f)
    
    # Pack gradients
    grads = d_U, d_V, d_W, d_b_hidden, d_b_out    
    
    # Clip gradients
    grads = clip_gradient_norm(grads)
    
    return loss, grads
#-------------------------------------------------------------------------

# print(f'targets shape: {test_target.shape}') # shape: (14, 4, 1)
# print(f'targets:\n{test_target}')
# print('-----')
# print(f'inputs shape: {test_input.shape}') # shape: (14, 4, 1)
# print(f'outputs:\n{outputs}')
# print('----------------------------------------------')
loss, grads = backward_pass(inputs = test_input, outputs = outputs, 
                            hidden_states = hidden_states, targets = test_target, params = params)

print('We get a loss of:')
print(loss)

##########################################################################
##
##  PART 9
##
##########################################################################
def update_parameters(params, grads, lr=1e-3):
    # Take a step
    for param, grad in zip(params, grads):
        param -= lr * grad
    
    return params
#-------------------------------------------------------------------------

import matplotlib.pyplot as plt
# %matplotlib inline

# Hyper-parameters
num_epochs = 1000

# Initialize a new network
params = init_rnn(hidden_size=hidden_size, vocab_size=vocab_size)

# Initialize hidden state as zeros
hidden_state = np.zeros((hidden_size, 1))

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
        hidden_state = np.zeros_like(hidden_state)

        # Forward pass
        outputs, hidden_states = forward_pass(
            inputs = inputs_one_hot, hidden_state = hidden_state, params = params
        )

        # Backward pass - returns loss and grads ( _ )
        loss, _ = backward_pass(inputs = inputs_one_hot, outputs = outputs, 
            hidden_states = hidden_states, targets = targets_one_hot, params = params
        )
        
        # Update loss
        epoch_validation_loss += loss
    
    # For each sentence in training set
    for inputs, targets in training_set:
        # One-hot encode input and target sequence
        inputs_one_hot = one_hot_encode_sequence(sequence = inputs, vocab_size = vocab_size, param_word_to_idx = word_to_idx)
        targets_one_hot = one_hot_encode_sequence(sequence = targets, vocab_size = vocab_size, param_word_to_idx = word_to_idx)
        
        # Re-initialize hidden state
        hidden_state = np.zeros_like(hidden_state)

        # Forward pass
        outputs, hidden_states = forward_pass(
            inputs = inputs_one_hot, hidden_state = hidden_state, params = params
        )

        # Backward pass
        loss, grads = backward_pass(
            inputs = inputs_one_hot, outputs = outputs, hidden_states = hidden_states, 
            targets = targets_one_hot, params = params
        )
        
        if np.isnan(loss): #not a number
            raise ValueError('Gradients have vanished!')
        
        # Update parameters
        params = update_parameters(params, grads, lr=3e-4)
        
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
hidden_state = np.zeros((hidden_size, 1))

# Forward pass
outputs, hidden_states = forward_pass(inputs_one_hot, hidden_state, params)
output_sentence = [idx_to_word[np.argmax(output)] for output in outputs]
print('Input sentence:')
print(inputs)

print('\nTarget sequence:')
print(targets)

print('\nPredicted sequence:')
print([idx_to_word[np.argmax(output)] for output in outputs])

# Plot training and validation loss
epoch = np.arange(len(training_loss))
plt.figure()
plt.plot(epoch, training_loss, 'r', label='Training loss',)
plt.plot(epoch, validation_loss, 'b', label='Validation loss')
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('NLL')
plt.show()