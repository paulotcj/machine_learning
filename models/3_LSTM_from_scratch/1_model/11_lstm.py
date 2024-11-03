import numpy as np
np.random.seed(42)
# original reference: https://github.com/CaptainE/RNN-LSTM-in-numpy
##########################################################################
##
##  PART 1
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
def execute_part1(hidden_layer_size, vocab_size):
    z_size = hidden_layer_size + vocab_size 

    params = init_lstm(hidden_size=hidden_layer_size, vocab_size=vocab_size, z_size=z_size)
#-------------------------------------------------------------------------
execute_part1(hidden_layer_size = 50, vocab_size = 4)

