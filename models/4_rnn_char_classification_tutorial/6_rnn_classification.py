
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
    files_pattern = 'data/names/*.txt'
    result_findFiles = findFiles(files_pattern)
    print(f'findFiles result:\n{result_findFiles}\n' )
    #----
    print(f'\nstring.ascii_letters:\n    {string.ascii_letters}')

    print('----')
    all_letters = string.ascii_letters + " .,;'"
    all_letters_idx = {letter: idx for idx, letter in enumerate(all_letters)}
    print(f'\nall_letters:\n    {all_letters}')
    print(f'all_letters_idx:\n    {all_letters_idx}')
    n_letters = len(all_letters)
    print(f'len(all_letters): {n_letters}\n')
    
    print('----')
    result_unicode_to_ascii = unicodeToAscii(all_letters=all_letters, input_str='Ślusàrski')
    print(f'unicodeToAscii(\'Ślusàrski\') : {result_unicode_to_ascii}')
    #----

    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []
    # print(f'result_findFiles: {result_findFiles}')
    
    for filename in result_findFiles:
        category = os.path.splitext(os.path.basename(filename))[0] # get the filename without the extension, e.g. 'Italian' from 'Italian.txt'
        all_categories.append(category) # list with the categories
        lines = readLines(filename, all_letters=all_letters)
        category_lines[category] = lines #dict with category as key and list of names as value
    #----
    n_categories = len(all_categories)
    print('\n\n----')
    print(f'len(all_categories): {n_categories}')
    print(f'all_categories: {all_categories}')
    print('----')
    # print('category_lines:')
    # for k , v in category_lines.items():
    #     print(f'{k}: {v}')
    # print(f'category_lines: {category_lines}')
    # print('----')
    print(f'category_lines[\'Italian\'][:5]   ->  {category_lines["Italian"][:5]  }')

    return {
        'n_letters': n_letters,
        'all_letters': all_letters,
        'all_letters_idx': all_letters_idx,
        'n_categories': n_categories,
        'all_categories': all_categories,
        'category_lines': category_lines
    }
#-------------------------------------------------------------------------

result_part1 = execute_part1()



##########################################################################
##
##  PART 1
##
##########################################################################
import torch
{
# #-------------------------------------------------------------------------
# def letterToIndex(letter, all_letters_idx): # Find letter index from all_letters, e.g. "a" = 0
#    
#     # return all_letters.find(letter)
#
#     return all_letters_idx[letter]
# #-------------------------------------------------------------------------
}
#-------------------------------------------------------------------------
def letterToTensor(letter, n_letters, all_letters_idx): # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    
    tensor = torch.zeros(1, n_letters) # fills the tensor with zeros, 1 row and n_letters columns
    # idx_of_letter = letterToIndex(letter = letter, all_letters_idx = all_letters_idx)
    idx_of_letter = all_letters_idx[letter]
    tensor[0][ idx_of_letter ] = 1
    
    return tensor
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def lineToTensor(line, n_letters, all_letters_idx): # Turn a line into a <line_length x 1 x n_letters>, or an array of one-hot letter vectors
    
    # fills the tensor with zeros - the one character tensor was 1 row and n_letters columns, now 
    #   considering we will have a line or word, we will have line len, 1 row, and n_letters columns
    tensor = torch.zeros(len(line), 1, n_letters)

    # for every letter in the line, we will get the index of the letter in the all_letters_idx dictionary
    #    and set the value of that index in the tensor to 1
    for li, letter in enumerate(line): 
        idx_of_letter = all_letters_idx[letter] #get the index of the letter

        # in the right position for the letter being investigated set the value of target index to 1
        tensor[li][0][ idx_of_letter ] = 1 
    
    return tensor
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part2(n_letters, all_letters_idx):

    result_letterToTensor = letterToTensor('a', n_letters = n_letters, all_letters_idx = all_letters_idx)
    print(f'letterToTensor(\'a\'): {result_letterToTensor}')

    result_letterToTensor = letterToTensor('b', n_letters = n_letters, all_letters_idx = all_letters_idx)
    print(f'letterToTensor(\'b\'): {result_letterToTensor}')    

    result_letterToTensor = letterToTensor('J', n_letters = n_letters, all_letters_idx = all_letters_idx)
    print(f'letterToTensor(\'J\'): {result_letterToTensor}')

    result_lineToTensor = lineToTensor(line = 'aa', n_letters = n_letters, all_letters_idx = all_letters_idx)
    print(f'lineToTensor(\'aa\').size(): {result_lineToTensor.size()}')
    print(f'lineToTensor(\'aa\'): {result_lineToTensor}')

    result_lineToTensor = lineToTensor(line = 'ab', n_letters = n_letters, all_letters_idx = all_letters_idx)    
    print(f'\n\nlineToTensor(\'ab\'): {result_lineToTensor}')


    result_lineToTensor = lineToTensor(line = 'Jones', n_letters = n_letters, all_letters_idx = all_letters_idx)
    print(f'lineToTensor(\'Jones\').size(): {result_lineToTensor.size()}')
    print(f'\n\nlineToTensor(\'Jones\'): {result_lineToTensor}')
#-------------------------------------------------------------------------
execute_part2(
    n_letters = result_part1['n_letters'], 
    all_letters_idx = result_part1['all_letters_idx']
    
)


##########################################################################
##
##  PART 3
##
##########################################################################

import torch.nn as nn
import torch.nn.functional as F




#-------------------------------------------------------------------------
class RNN(nn.Module):
    #-------------------------------------------------------------------------
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(RNN, self).__init__() #can use super().__init__()


        self.hidden_size = hidden_layer_size

        self.i2h = nn.Linear(input_size, hidden_layer_size) # i2h -> input to hidden, transforms the input to the hidden layer
        self.h2h = nn.Linear(hidden_layer_size, hidden_layer_size) # h2h -> hidden to hidden, transforms the hidden state from the previous time step to the hidden state of the current time step
        self.h2o = nn.Linear(hidden_layer_size, output_size) # h2o -> hidden to output, ransforms the hidden state to the output
        self.softmax = nn.LogSoftmax(dim=1) # softmax layer
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    def forward(self, input, hidden):
        # The hidden parameter is passed into the method, representing the hidden state from the 
        # previous time step.
        # The new hidden state is computed using both the current input and the previous 
        #   hidden state: hidden = F.tanh(self.i2h(input) + self.h2h(hidden)).
        # The method returns both the output and the updated hidden state: return output, hidden.
        # This shows that the hidden state is maintained and updated at each time step, and then 
        #   passed on to the next time step, which is a defining feature of Recurrent Neural Networks.
        #
        # The structure of the forward method processes sequences of inputs one step at a time, 
        #   updating the hidden state at each step in a classic implementation of RNNs.

        temp = self.i2h(input) + self.h2h(hidden)
        hidden = F.tanh(temp)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part3(n_letters, n_categories, all_letters_idx):

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Torch: Running on GPU")
    else:
        device = torch.device('cpu')
        print("Torch: Running on CPU")


    # remember categories are the languages, in our example typically 18
    #  and the number of letters also typically 57
    n_hidden = 128


    rnn = RNN(input_size=n_letters, hidden_layer_size=n_hidden, output_size=n_categories) # 57, 128, 18

    input = letterToTensor(letter='A', n_letters=n_letters, all_letters_idx=all_letters_idx)
    hidden = torch.zeros(1, n_hidden)

    # When you use parentheses () on an instance of a class, it triggers the __call__ method of that class
    # In most neural network libraries like PyTorch, the __call__ method is typically overridden 
    #   to call the forward method of the model
    output, next_hidden = rnn(input, hidden)


    input = lineToTensor(line = 'Albert', n_letters=n_letters, all_letters_idx=all_letters_idx)
    hidden = torch.zeros(1, n_hidden)

    print('\n\nThe line below is a tensor with random values so far')
    output, next_hidden = rnn(input[0], hidden)
    print(output) 

    return {
        'output': output,
        'rnn': rnn
    }   
#-------------------------------------------------------------------------

result_part3 = execute_part3(
    n_letters = result_part1['n_letters'], 
    n_categories = result_part1['n_categories'],
    all_letters_idx = result_part1['all_letters_idx']
)


##########################################################################
##
##  PART 4
##
##########################################################################
import random

#-------------------------------------------------------------------------
def categoryFromOutput(all_categories, output):
    # remember categories are the languages, in our example typically 18
    #  and the number of letters also typically 57

    # output is length 18, and the top_n is the value of the highest output, meaning the one the RNN
    # thinks is the most likely (language), and top_i is the index of that value
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item() #extract the index from top_i tensor
    # print('----')
    # print(f'output: {output}')
    # print(f'top_n: {top_n}')
    # print(f'top_i: {top_i}')
    # print(f'category_i: {category_i}')
    # print('----')
    return all_categories[category_i], category_i #return the language and the index
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def randomChoice(list_input):

    #return a random element from the list using a random module to pick an index from 0 to the length-1 of the list
    return list_input[ random.randint(0, len(list_input) - 1) ]
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def randomTrainingExample(all_categories, category_lines, n_letters, all_letters_idx):
    # picks 1 category (language) and 1 name from that category, and return the tensor for
    #   the category and the tensor for the name


    category = randomChoice(all_categories) #pick an item from the list categories, e.g. 'Italian'
    # for k , v in category_lines.items():
    #     print(f'category key: {k}')
    #     for i in v:
    #         print(f'    {i}')

    # given a string category (e.g. 'Italian') we will pick a random name from the list of names belonging
    #   to that category
    line = randomChoice(category_lines[category])

    index_of_category = all_categories.index(category) # from all categories, find the index of the category string provided
    category_tensor = torch.tensor([index_of_category], dtype=torch.long) #just wrap the index in a tensor, so if index was 3 it will be tensor([3])

    # print(f'index_of_category: {index_of_category}')
    # print(f'category_tensor: {category_tensor}')


    line_tensor = lineToTensor(line = line, n_letters = n_letters, all_letters_idx = all_letters_idx)

    return category, line, category_tensor, line_tensor
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part4(all_categories, output, category_lines, n_letters, all_letters_idx):
    
    result_categoryFromOutput = categoryFromOutput(all_categories = all_categories, output = output)
    print(f'result_categoryFromOutput:\n{result_categoryFromOutput}\n\n')

    for i in range(10):
        category, line, category_tensor, line_tensor = randomTrainingExample(
            all_categories = all_categories, 
            category_lines = category_lines,
            n_letters = n_letters,
            all_letters_idx = all_letters_idx
        )
        print(f'category(language) = {category}\t- line (name)={line}')    

#-------------------------------------------------------------------------
execute_part4( 
    all_categories  = result_part1['all_categories'],
    output          = result_part3['output'],
    category_lines  = result_part1['category_lines'],
    n_letters       = result_part1['n_letters'],
    all_letters_idx = result_part1['all_letters_idx']
)


##########################################################################
##
##  PART 5
##
##########################################################################
import time
import math
#-------------------------------------------------------------------------
def train(rnn, learning_rate, category_tensor, line_tensor):
    hidden_state = rnn.initHidden() #initialize the hidden state to zeros

    # The nn.NLLLoss() function stands for 'Negative Log Likelihood Loss'
    #   which is commonly used in classification problems, particularly when 
    #   the model's output is a log-probability distribution
    criterion = nn.NLLLoss()       

    # reset the gradients of all the parameters in the rnn model to zero. During the backward 
    #   pass of training, gradients are accumulated into the .grad attributes of each parameter 
    #   If you do not reset these gradients, they will accumulate across multiple trainings
    rnn.zero_grad()

    # remember: line_tensor is the name, it's the name's length x 1 x n_letters (one-hot encoding)
    #   so a name of length 7 would be [7,1,57] (57 being len of all letters)
    #   category_tensor is the language, it's just a tensor with the index of the language

    # print(f'line_tensor: {line_tensor}')
    # print(f'line_tensor[0]: {line_tensor[0]}')

    #so for tensor line/name of length 7, we would have this tensor: [7,1,57], therefore looping 7 times
    for i in range(line_tensor.size()[0]):
        # get the predicted output and the hidden state for the current character in the name
        #  note that only the last output is used for the loss calculation
        output, hidden_state = rnn(line_tensor[i], hidden_state) #feed one tensor at time

    loss = criterion(output, category_tensor)
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

    """
    Explanation of the code above: 
        p.data is the raw data of the tensor
        p.grad.data is the raw data of the gradient tensor
        add_ is an in-place addition operation. (in-place denoted by '_') 
        alpha=-learning_rate specifies the scaling factor for the addition

    For instance: If learning_rate is 0.01, and a parameter p has a gradient of 0.1, the update 
      would be: p.data = p.data - learning_rate * gradient -> p.data = p.data - 0.01 * 0.1
    """

    return output, loss.item()    
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
def execute_train_rnn(rnn, n_iters, all_categories, category_lines, n_letters, all_letters_idx, 
        learning_rate, print_every, plot_every):
    
 

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []


    start = time.time()

    #----------------
    for iter in range(1, n_iters + 1):
        #----
        #fetch a random training example
        category, line, category_tensor, line_tensor = randomTrainingExample(
            all_categories  = all_categories,
            category_lines  = category_lines,
            n_letters       = n_letters,
            all_letters_idx = all_letters_idx
        )
        #----
        # train on that example
        output, loss = train(
            rnn             = rnn, 
            learning_rate   = learning_rate,
            category_tensor = category_tensor, 
            line_tensor     = line_tensor
        )
        #----
        
        current_loss += loss

        #----------------
        # Print summary of training
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(all_categories = all_categories, output = output)

            correct = '✓' if guess == category else f'✗ ({category})'

            training_progress = (iter / n_iters * 100)
            time_since_start = timeSince(start)
            str_output = f'{iter}\t{training_progress:.0f}% ({time_since_start})\tloss:{loss:.4f}\tName: {line} - Class:{guess} {correct}'

            print(str_output)
        #----------------

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every) # makes an average of the loss between the plot_every iterations
            current_loss = 0
        #----------------
    #----------------
    return {
        'all_losses': all_losses
    }
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part5(rnn, all_categories, category_lines, n_letters, all_letters_idx):

    print('\n\n----------\nTraining:') # give some clear separation for the training output

    learning_rate = 0.005 # 0.005 is typical for RNNs, higher may lead to exploding gradients, lower to vanishing gradients

    n_iters = 100_000
    print_every = 5_000
    plot_every = 1_000

    result_execute_train = execute_train_rnn(
        rnn             = rnn, 
        n_iters         = n_iters, 
        all_categories  = all_categories,
        category_lines  = category_lines,
        n_letters       = n_letters,
        all_letters_idx = all_letters_idx,
        learning_rate   = learning_rate,
        print_every     = print_every, 
        plot_every      = plot_every
    )

    return {
        'all_losses': result_execute_train['all_losses']
    }
#-------------------------------------------------------------------------

result_part5 = execute_part5(
    rnn = result_part3['rnn'],
    all_categories = result_part1['all_categories'],
    category_lines = result_part1['category_lines'],
    n_letters = result_part1['n_letters'],
    all_letters_idx = result_part1['all_letters_idx']
)


##########################################################################
##
##  PART 6
##
##########################################################################


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#-------------------------------------------------------------------------
def show_graph(all_losses):
    plt.figure()
    plt.plot(all_losses)
    plt.show()
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def make_prediction(rnn, line_tensor): # Just return an output given a line - used in the confusion matrix
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def show_confusion_matrix(rnn, n_categories, all_categories, category_lines, n_letters, all_letters_idx):
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)

    #------------------
    # Go through a bunch of examples and record which are correctly guessed
    n_confusion = 10_000
    for i in range(n_confusion):
        #----
        # 1 -get the data to verify
        category, line, category_tensor, line_tensor = randomTrainingExample(
            all_categories  = all_categories, 
            category_lines  = category_lines, 
            n_letters       = n_letters, 
            all_letters_idx = all_letters_idx
        )
        #----

        # 2 - get the actual index of the category generated above
        actual_category_idx = all_categories.index(category)        
        
        # 3 - make a prediction
        output = make_prediction(rnn = rnn,line_tensor = line_tensor)

        # 4 - from the prediction get the category and the index of the category
        category_prediction, category_prediction_idx = categoryFromOutput(
            all_categories  = all_categories, 
            output          = output
        )

        # 5 - In a matix of size [n_categories x n_categories] set to +1 where the coordinates match
        #   actual_category_idx vs category_prediction_idx. Note that ideally if the prediction is correct
        #   then actual_category_idx must be equal to category_prediction_idx. However, in some cases this
        #   will not happen (due to incorrect predictions). And as we run this many times over, we should
        #   expect to see a transversal line in the matrix with 'higher' values, and everything else with
        #   lower values or close to zero - that is, if our RNN was trained properly.
        confusion[actual_category_idx][category_prediction_idx] += 1
    #------------------

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        # this is a bit confusing, but we get the sum() from all elements in that row, and then
        #   we divide each element in that row by the sum. So this expression here is a bit like:
        #   confusion[i].every_and_each / confusion[i].sum() - and the se set this as a row in the matrix
        #   For instance: row = [1,2,3], sum = 6, then row = [1/6, 2/6, 3/6] -> [0.16666667, 0.33333333, 0.5]
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part6(all_losses):
    show_graph(all_losses)
    show_confusion_matrix(
        rnn             = result_part3['rnn'], 
        n_categories    = result_part1['n_categories'], 
        all_categories  = result_part1['all_categories'],
        category_lines  = result_part1['category_lines'],
        n_letters       = result_part1['n_letters'],
        all_letters_idx = result_part1['all_letters_idx']
    )
#-------------------------------------------------------------------------
execute_part6(all_losses = result_part5['all_losses'])























