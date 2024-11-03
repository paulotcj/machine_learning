
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

    print('# categories:', n_categories, all_categories)
    print(unicodeToAscii("O'Néàl"))         
#-------------------------------------------------------------------------
execute_part1()


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
    def __init__(self, input_size, hidden_size, output_size, n_categories):
        super(RNN, self).__init__() # can use super().__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size) # i2h -> input to hidden, transforms the input to the hidden layer
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size) # h2h -> hidden to hidden, transforms the hidden state from the previous time step to the hidden state of the current time step
        self.o2o = nn.Linear(hidden_size + output_size, output_size) # h2o -> hidden to output, ransforms the hidden state to the output
        
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
    def forward(self, category, input, hidden):

        input_combined = torch.cat((category, input, hidden), 1) # concatenates the: category, input, and hidden tensors, along the second dimension (dim=1), this combined tensor will be used as the input for the next layers
        hidden = self.i2h(input_combined) # input-to_hidden: the combined input tensor is passed through a linear layer (i2h) to compute the new hidden state - this layer transforms the input to the hidden layer
        output = self.i2o(input_combined) # intput-to-output: same as above (i2h), but this layer transforms the input to the output layer
        
        output_combined = torch.cat((hidden, output), 1) # the newly computed hidden state and output are concatenated along the second dimension to form a new combined tensor
        
        output = self.o2o(output_combined) # output-to_output: this combined tensor is passed through another linear layer (o2o) to compute the final output

        output = self.dropout(output) # apply dropout
        output = self.softmax(output) # apply softmax: convert the raw scores into log-probabilities
        return output, hidden #return the output and the hidden state (for the next time step)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def randomChoice(list_input):
    #return a random element from the list using a random module to pick an index from 0 to the length-1 of the list
    return list_input[ random.randint(0, len(list_input) - 1) ]
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# Get a random category and random line from that category
def randomTrainingPair(all_categories, category_lines):

    category = randomChoice(all_categories) #get a category
    line = randomChoice(category_lines[category]) # and then from that category, get a random line/name
    return category, line
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# One-hot vector for category
def categoryTensor(category, all_categories, n_categories):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line, all_letters, n_letters):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# ``LongTensor`` of second letter to end (EOS) for target
def targetTensor(line, all_letters, n_letters):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample(all_categories, category_lines, n_categories, all_letters, n_letters):

    category, line = randomTrainingPair(
        all_categories = all_categories, 
        category_lines = category_lines
    )

    category_tensor = categoryTensor(
        category        = category, 
        all_categories  = all_categories, 
        n_categories    = n_categories
    )

    input_line_tensor = inputTensor(
        line        = line, 
        all_letters = all_letters, 
        n_letters   = n_letters
    )

    target_line_tensor = targetTensor(
        line        = line, 
        all_letters = all_letters, 
        n_letters   = n_letters
    )

    return category_tensor, input_line_tensor, target_line_tensor
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def train(rnn, category_tensor, input_line_tensor, target_line_tensor):

    criterion = nn.NLLLoss()
    learning_rate = 0.0005

    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = torch.Tensor([0]) # you can also just simply use ``loss = 0``

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)
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
def execute_train(all_letters, n_letters, all_categories, n_categories, category_lines):
    rnn = RNN(input_size = n_letters, hidden_size = 128, output_size=n_letters, n_categories=n_categories)

    n_iters = 100000
    print_every = 5000
    plot_every = 500
    all_losses = []
    total_loss = 0 # Reset every ``plot_every`` ``iters``

    start = time.time()

    #-----------------------
    for iter in range(1, n_iters + 1):
        output, loss = train(*randomTrainingExample(all_categories=all_categories, category_lines=category_lines, n_categories=n_categories, all_letters=all_letters, n_letters=n_letters))
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0
    #-----------------------

#-------------------------------------------------------------------------
        