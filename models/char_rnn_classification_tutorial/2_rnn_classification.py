
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
    findFiles_result = findFiles(files_pattern)
    print(f'findFiles result:\n{findFiles_result}\n' )
    #----
    print(f'\nstring.ascii_letters:\n    {string.ascii_letters}')

    print('----')
    all_letters = string.ascii_letters + " .,;'"
    print(f'\nall_letters:\n    {all_letters}')
    n_letters = len(all_letters)
    print(f'len(all_letters): {n_letters}\n')
    
    print('----')
    unicode_to_ascii_result = unicodeToAscii(all_letters=all_letters, input_str='Ślusàrski')
    print(f'unicodeToAscii(\'Ślusàrski\') : {unicode_to_ascii_result}')
    #----

    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []
    # print(f'findFiles_result: {findFiles_result}')
    
    for filename in findFiles_result:
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
    }
#-------------------------------------------------------------------------

execute1_result = execute_part1()



##########################################################################
##
##  PART 1
##
##########################################################################
import torch
#-------------------------------------------------------------------------
def letterToIndex(letter, all_letters): # Find letter index from all_letters, e.g. "a" = 0
    
    return all_letters.find(letter)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def letterToTensor(letter, n_letters, all_letters): # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    
    tensor = torch.zeros(1, n_letters) # fills the tensor with zeros, 1 row and n_letters columns
    tensor[0][letterToIndex(letter = letter, all_letters = all_letters)] = 1
    
    return tensor
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def lineToTensor(line, n_letters, all_leters): # Turn a line into a <line_length x 1 x n_letters>, or an array of one-hot letter vectors
    
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter = letter, all_letters = all_leters)] = 1
    
    return tensor
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part2(n_letters, all_letters):

    letterToTensor_result = letterToTensor('J', n_letters = n_letters, all_letters = all_letters)
    print(f'letterToTensor(\'J\'): {letterToTensor_result}')

    lineToTensor_result = lineToTensor(line = 'Jones', n_letters = n_letters, all_leters = all_letters)
    print(f'lineToTensor(\'Jones\').size(): {lineToTensor_result.size()}')
#-------------------------------------------------------------------------
execute_part2(
    n_letters = execute1_result['n_letters'], 
    all_letters = execute1_result['all_letters']
)