
##########################################################################
##
##  IMPORTS
##
##########################################################################

from io import open
import glob
import os
import unicodedata
import string

##########################################################################
##
##  PART 1
##
##########################################################################
#-------------------------------------------------------------------------
def findFiles(path): #returns an array with the names of the files in the directory
    return glob.glob(path) # glob is used to retrieve files and directories that match a specified pattern
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def unicodeToAscii(all_letters, input_str): # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427

    # unicodedata.normalize('NFD', input_str) -> normalize Unicode strings. Unicode normalization is 
    #   a process that converts text to a standard form, which can be useful for string comparison, 
    #   searching, and other text processing tasks.
    # In this specific call, the function normalize is being used with the normalization form 'NFD'. 
    #   The 'NFD' stands for Normalization Form D (Canonical Decomposition). This form decomposes combined 
    #   characters into their constituent parts. For example, a character like 'é' 
    #   (which is a single character) would be decomposed into 'e' and an accent character.
    #
    # if unicodedata.category(c) != 'Mn' -> In this specific case, the code is checking if the category of 
    #   the character c is not equal to 'Mn'. The category 'Mn' stands for "Mark, Nonspacing". Nonspacing 
    #   marks are characters that typically combine with preceding characters and do not occupy a space 
    #   by themselves, such as accents or diacritical marks.
    #   By using this condition, the code is likely filtering out nonspacing marks from further processing. 
    #   This can be useful in text processing tasks where you want to ignore or remove diacritical marks 
    #   to simplify the text or to perform operations that are sensitive to such marks. For example, 
    #   in text normalization or in preparing text for machine learning models, it might be necessary to 
    #   strip out these marks to ensure consistency and accuracy.

    temp = [
        c for c in unicodedata.normalize('NFD', input_str)
           if unicodedata.category(c) != 'Mn' and 
           c in all_letters
    ]
    
    return ''.join(temp)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def readLines(filename, all_letters): # Read a file and split into lines
    # .strip -> removes any leading and trailing whitespace characters
    lines = open(filename, encoding='utf-8').read().strip().split('\n')

    # for every line (name) we will convert it to ASCII, and return a list with those names
    return [unicodeToAscii(all_letters=all_letters, input_str=line) for line in lines] 
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part1():

    all_letters = string.ascii_letters + " .,;'-"
    all_letters_idx = {letter: idx for idx, letter in enumerate(all_letters)}
    n_letters = len(all_letters) + 1 # Plus EOS marker

    files_pattern = 'data/names/*.txt'
    result_findFiles = findFiles(files_pattern)
    print(f'findFiles result:\n{result_findFiles}\n' )
    #----
    category_lines = {}
    all_categories = []

    #-----------------------------------------
    # Build the category_lines dictionary, a list of lines per category
    for filename in findFiles(files_pattern):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename = filename, all_letters=all_letters)
        category_lines[category] = lines
    #-----------------------------------------    
    n_categories = len(all_categories)

    if n_categories == 0:
        raise RuntimeError('Data not found. Make sure that you downloaded data '
            'from https://download.pytorch.org/tutorial/data.zip and extract it to '
            'the current directory.')   

    print(f'Number of categories: {n_categories}')
    print(f'All categories: {all_categories}')

    name_test_str = "O'Néàl"
    result_unicodeToAscii = unicodeToAscii( all_letters=all_letters, input_str = name_test_str)
    print(f'\n\nUnicode to ASCII - Original string: {name_test_str} -> Result: {result_unicodeToAscii}')  

    return {
        'all_letters'       : all_letters,
        'n_letters'         : n_letters,
        'all_categories'    : all_categories,
        'n_categories'      : n_categories,
        'category_lines'    : category_lines,
        'all_letters_idx'   : all_letters_idx
    }
#-------------------------------------------------------------------------
result_part1 = execute_part1()
##########################################################################
##
##  PART 2
##
##########################################################################
import torch
import torch.nn as nn
import random
import time
import math
#-------------------------------------------------------------------------
class RNN(nn.Module):
    #-------------------------------------------------------------------------
    def __init__(self, input_size, hidden_layer_size, output_size, n_categories):
        super(RNN, self).__init__() # can use super().__init__()
        self.hidden_size = hidden_layer_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_layer_size, hidden_layer_size) # i2h -> input to hidden, transforms the input to the hidden layer
        self.i2o = nn.Linear(n_categories + input_size + hidden_layer_size, output_size) # h2h -> hidden to hidden, transforms the hidden state from the previous time step to the hidden state of the current time step
        self.o2o = nn.Linear(hidden_layer_size + output_size, output_size) # h2o -> hidden to output, ransforms the hidden state to the output
        
        #--------
        # Dropout is a regularization technique used to prevent overfitting in neural networks 
        #   by randomly setting a fraction of input units to zero at each update during training 
        #   time. This helps the model to generalize better by reducing the reliance on specific 
        #   neurons and encouraging the network to learn more robust features
        # In this specific case, nn.Dropout(0.1) means that during each forward pass, there is 
        #   a 10% chance that any given element in the input tensor will be zeroed out. The 
        #   remaining 90% of the elements will be scaled by a factor of approx 1.11 (1/0.9) 
        #   to maintain the expected sum of the inputs.         
        self.dropout = nn.Dropout(0.1)
        #--------
        
        self.softmax = nn.LogSoftmax(dim=1) # softmax layer
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, category, input, hidden_state):

        input_combined = torch.cat((category, input, hidden_state), 1) # concatenates the: category, input, and hidden tensors, along the second dimension (dim=1), this combined tensor will be used as the input for the next layers
        hidden_state = self.i2h(input_combined) # input-to_hidden: the combined input tensor is passed through a linear layer (i2h) to compute the new hidden state - this layer transforms the input to the hidden layer
        output = self.i2o(input_combined) # intput-to-output: same as above (i2h), but this layer transforms the input to the output layer
        
        output_combined = torch.cat((hidden_state, output), 1) # the newly computed hidden state and output are concatenated along the second dimension to form a new combined tensor
        
        output = self.o2o(output_combined) # output-to_output: this combined tensor is passed through another linear layer (o2o) to compute the final output

        output = self.dropout(output) # apply dropout
        output = self.softmax(output) # apply softmax: convert the raw scores into log-probabilities
        return output, hidden_state #return the output and the hidden state (for the next time step)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def timeSince(since):
    now = time.time() #float
    time_diff = now - since #float
    minutes = math.floor(time_diff / 60) #int
    seconds = math.floor(time_diff - minutes * 60) #float
    
    return f'{minutes}m {seconds}s'
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def randomChoice(list_input):
    #return a random element from the list using a random module to pick an index from 0 to the length-1 of the list
    return list_input[ random.randint(0, len(list_input) - 1) ]
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def randomTrainingPair(all_categories, category_lines): # Get a random category and random line from that category

    category = randomChoice(all_categories) #get a category
    line = randomChoice(category_lines[category]) # and then from that category, get a random line/name
    return category, line
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def categoryTensor(category, all_categories, n_categories): # One-hot vector for category
    category_idx = all_categories.index(category)
    return_tensor = torch.zeros(1, n_categories) # [1, n_categories]
    return_tensor[0][category_idx] = 1 # [0, category_idx] = 1 , so we could expect something like [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    return return_tensor
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def inputTensor(line, n_letters, all_letters_idx): # One-hot matrix of first to last letters (not including EOS) for input
    len_line = len(line)

    tensor = torch.zeros(len_line, 1, n_letters) # [len_line, 1, n_letters] -> for every line/name we will have 1 row and n_letters columns

    # for every letter in the line, we will get the index of the letter in the all_letters_idx dictionary
    #    and set the value of that index in the tensor to 1
    for li, letter in enumerate(line): 
        idx_of_letter = all_letters_idx[letter] #get the index of the letter

        tensor[li][0][idx_of_letter] = 1 # in the right position for the letter being investigated set the value of target index to 1

    
    return tensor
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def targetTensor(line, n_letters, all_letters_idx): # ``LongTensor`` of second letter to end (EOS) for target
    len_line = len(line)

    letter_indexes = [
        all_letters_idx[line[li]]
        for li in range(1, len_line)
    ]

    
    # Note: We defined the letter vocab, but never included the EOS token in the vocab. But we made an adjusment so the 
    #   n_letters = len(all_letters), and then we added 1 to n_letters, so the EOS token will be included in the vocab represented by
    #   the index n_letters - 1
    eos_letter = n_letters - 1
    letter_indexes.append(eos_letter) # EOS

    return torch.LongTensor(letter_indexes)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def randomTrainingExample(all_categories, category_lines, n_categories, n_letters, all_letters_idx): # Make category, input, and target tensors from a random category, line pair
    
    # output a string for each, e.g.: category -> 'Irish', line -> 'Brady'
    category, line = randomTrainingPair(
        all_categories = all_categories, 
        category_lines = category_lines
    )

    # transform the category above into a tensor, e.g.: tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    category_tensor = categoryTensor(
        category        = category, 
        all_categories  = all_categories, 
        n_categories    = n_categories
    )

    # transform the line into a tensor, e.g.: tensor([[[0, 0, 0, ..., 0, 0, 0]]])
    input_line_tensor = inputTensor(
        line            = line, 
        n_letters       = n_letters,
        all_letters_idx = all_letters_idx
    )

    target_line_tensor = targetTensor(
        line            = line, 
        n_letters       = n_letters,
        all_letters_idx = all_letters_idx, 
    )


    # print(f'category_tensor: {category_tensor}')
    # print(f'input_line_tensor: {input_line_tensor}')
    # print(f'target_line_tensor: {target_line_tensor}')


    return category_tensor, input_line_tensor, target_line_tensor
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def train(rnn, category_tensor, input_line_tensor, target_line_tensor):

    criterion = nn.NLLLoss()
    learning_rate = 0.0005

    # in-place operation adding a dimension of size 1 at the position specified. -1 tells that the dimension will be added at the last 
    #   position of the tensor. Typically we would have a tensor shape of [4], and then it would become [4,1], 
    #   e.g.: tensor([ 0, 17, 17, 14, 18, 58]) -> tensor([[ 0], [17], [17], [14], [18], [58]])
    target_line_tensor.unsqueeze_(-1)


    hidden_state = rnn.initHidden() #hidden state initialized to zeros

    rnn.zero_grad()

    loss = torch.Tensor([0]) # you can also just simply use ``loss = 0``

    #--------------------------
    # for tensor line/name of length 7, we would have this tensor: [7,1,59], therefore looping 7 times
    for i in range(input_line_tensor.size(0)):
        
        output, hidden_state = rnn(category = category_tensor, input= input_line_tensor[i], hidden_state = hidden_state)

        curr_loss = criterion(output, target_line_tensor[i])

        loss += curr_loss
    #--------------------------

    
    loss.backward()
    """
    The way things are set up here is not very intuitive, so a more vanilla approach might shed a light
    in why we are doing this. Consider this example:

        # Instantiate the model, loss function, and optimizer
        model = SimpleRNN(input_size=10, hidden_size=20, output_size=1)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Dummy input and target
        input_data = torch.randn(5, 3, 10)  # (sequence_length, batch_size, input_size)
        target = torch.randn(3, 1)  # (batch_size, output_size)

        # Forward pass
        output = model(input_data)
        loss = criterion(output, target)    
    """    

    # Add parameters' gradients to their values, multiplied by learning rate
    #  paramaters = weights and biases
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    # remember that input_line_tensor.size(0) is the length of the line/name (example of tensor shape: [6,1,59])
    #  then we can have loss.item() = 24.39 and input_line_tensor.size(0) = 6, with return_loss = 4.065
    #  in other words we are returning the average loss per character
    return_loss = loss.item() / input_line_tensor.size(0)

    return output, return_loss
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_train(all_letters, n_letters, all_categories, n_categories, category_lines, all_letters_idx):
    rnn = RNN(
        input_size          = n_letters, 
        hidden_layer_size   = 128, 
        output_size         = n_letters, 
        n_categories        = n_categories
    )

    n_iters = 100_000
    print_every = 5_000
    plot_every = 500
    all_losses = []
    total_loss = 0 # Reset every ``plot_every`` ``iters``

    start = time.time()

    #-----------------------
    for iter in range(1, n_iters + 1):
        #--------
        category_tensor, input_line_tensor, target_line_tensor = randomTrainingExample(
            all_categories  = all_categories, 
            category_lines  = category_lines, 
            n_categories    = n_categories, 
            n_letters       = n_letters,
            all_letters_idx = all_letters_idx
        )



        output, loss = train(
            rnn                 = rnn, 
            category_tensor     = category_tensor, 
            input_line_tensor   = input_line_tensor, 
            target_line_tensor  = target_line_tensor
        )
        #--------

        total_loss += loss

        if iter % print_every == 0:
            time_elapsed = timeSince(start)
            percent_complete = math.floor(iter / n_iters * 100)
            print(f'time elapsed: {time_elapsed}\tpercent completed: {percent_complete}%\tloss: {loss}')
            # print('%s (%d %d%%) %.4f' % (time_elapsed, iter, iter / n_iters * 100, loss))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every) # makes an average of the loss between the plot_every iterations
            total_loss = 0
    #-----------------------

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part2(all_letters, n_letters, all_categories, n_categories, category_lines, all_letters_idx):

    execute_train(
        all_letters     = all_letters, 
        n_letters       = n_letters, 
        all_categories  = all_categories, 
        n_categories    = n_categories, 
        category_lines  = category_lines,
        all_letters_idx = all_letters_idx
    )
#-------------------------------------------------------------------------
execute_part2(
    all_letters     = result_part1['all_letters'], 
    n_letters       = result_part1['n_letters'], 
    all_categories  = result_part1['all_categories'], 
    n_categories    = result_part1['n_categories'], 
    category_lines  = result_part1['category_lines'],
    all_letters_idx = result_part1['all_letters_idx']
)


##########################################################################
##
##  PART 3
##
##########################################################################
import matplotlib.pyplot as plt
def execute_part3(all_losses):

    plt.figure()
    plt.plot(all_losses)