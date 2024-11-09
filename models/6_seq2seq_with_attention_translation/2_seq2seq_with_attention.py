##########################################################################
##
##  IMPORTS
##
##########################################################################

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import time
import math


##########################################################################
##
##  PART 1
##
##########################################################################

#-------------------------------------------------------------------------
class Lang:
    #-------------------------------------------------------------------------
    def __init__(self, name):
        self.name = name
        self.word2index = {} # dictionaries
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"} # first words, SOS: Start Of Sentence, EOS: End Of Sentence
        self.n_words = 2  # Count SOS and EOS
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def addSentence(self, sentence): # get a sentence and try to add a word to the dictionary
        for word in sentence.split(' '):
            self.addWord(word)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words # assume the index of the word dict length, we will add 1 to this count soon
            self.word2count[word] = 1 # word count, first occurrence
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def unicodeToAscii(input_str):

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

    return_str = ''.join(
        c for c in unicodedata.normalize('NFD', input_str)
        if unicodedata.category(c) != 'Mn'
    )

    return return_str
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def normalizeString(param_str): # Lowercase, trim, and remove non-letter characters, also separate punctuation from words by addint 1 space

    param_str = unicodeToAscii( param_str.lower().strip() ) # convert to ASCII, lowercase, and strip leading/trailing whitespaces

    # insert a space before every period, exclamation mark, or question mark. ensure that 
    #   punctuation is separated from words by a space, which helpful with text analysis
    param_str = re.sub(r"([.!?])", r" \1", param_str)

    # matches any sequence of one or more characters that are not letters, exclamation marks (!), 
    #   or question marks (?) and replaces them with a single space
    param_str = re.sub(r"[^a-zA-Z!?]+", r" ", param_str)

    return param_str.strip()    
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    file_name = f'data/{lang1}-{lang2}.txt' # most likely: eng-fra.txt
    # Read the file and split into lines
    full_file = open(file_name, encoding='utf-8').read().strip().split('\n')

    # this will be a list of lists, the list len is the number of lines in the file, and the
    #  list[x] length is 2


    sentences_pairs = [
        [
            normalizeString(sentence) 
            for sentence in line.split('\t') # 2 - get each version - the translations are separated by a tab
        ] 
        for line in full_file # 1 - get a line
    ]


    # Reverse pairs, make Lang instances
    if reverse:
        print(f'Reversing the order of the words in the sentences_pairs list. Reverse: {reverse}')
        # reverse the order of the words in the sentences_pairs list
        sentences_pairs = [ list(reversed(pair)) # reverse
                 for pair in sentences_pairs #from each line in the sentences_pairs list
        ]

        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        print(f'Keeping the order of the words in the sentences_pairs list. Reverse: {reverse}')
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

      

    return input_lang, output_lang, sentences_pairs
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def filterPair(pair, lang_prefixes, max_length = 10):

    # filter and filterPairs work together, the intention is to filter out long sentences, typical
    #   max len of 10 words, and also to filter out sentences that do not start with a specific 
    #   prefix - this prefix is a list of typical 'simple' senteces like 'I am sorry'

    # with that in mind, if the length of the first sentence is less than 10, and the 
    #   length of the second sentence is less than 10, and the second sentence starts with

    line_lang1 = pair[0].split(' ')
    len_line_lang1 = len(line_lang1)

    line_lang2 = pair[1].split(' ')
    len_line_lang2 = len(line_lang2)

    return_result = len_line_lang1 < max_length and \
                    len_line_lang2 < max_length and \
                    pair[1].startswith(lang_prefixes)

    return return_result
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def filterPairs(sentences_pairs, lang_prefixes):

    return [
        pair 
        for pair in sentences_pairs 
        if filterPair(pair = pair, lang_prefixes = lang_prefixes) #filter out long sentences and only pick based on the criteria from the prefixes
    ]
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def prepareData(lang1, lang2, lang1_prefixes, reverse=False):

    input_lang, output_lang, sentences_pairs = readLangs(lang1 = lang1, lang2=lang2, reverse = reverse)

    print("Read %s sentence pairs" % len(sentences_pairs))
    

    sentences_pairs = filterPairs(
        sentences_pairs = sentences_pairs, 
        lang_prefixes   = lang1_prefixes
    )


    print("Trimmed to %s sentence pairs" % len(sentences_pairs))

    print("Counting words...")
    for pair in sentences_pairs:
        #remember these are methods from the Lang class, and they count unique words
        input_lang.addSentence(pair[0]) 
        output_lang.addSentence(pair[1])

    print("Counted words:")
    print(f'Language: {input_lang.name} - Number of unique words: {input_lang.n_words}')
    print(f'Language: {output_lang.name} - Number of unique words: {output_lang.n_words}')


    return input_lang, output_lang, sentences_pairs
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part1():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print('Using GPU')
    else:
        print('Using CPU')

    SOS_token = 0
    EOS_token = 1

    MAX_LENGTH = 10

    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )    

    input_lang, output_lang, pairs = prepareData(
        lang1           = 'eng', 
        lang2           = 'fra', 
        lang1_prefixes  = eng_prefixes,  
        reverse         = True
    )
    print(random.choice(pairs))

    return {
        'device': device,
        'eos_token': EOS_token,
        'sos_token': SOS_token,
        'max_length': MAX_LENGTH,
        'lang_prefixes': eng_prefixes
    }
#-------------------------------------------------------------------------
result_part1 = execute_part1()


##########################################################################
##
##  PART 2
##
##########################################################################

#-------------------------------------------------------------------------
class EncoderRNN(nn.Module):
    #-------------------------------------------------------------------------
    def __init__(self, input_size, hidden_layer_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__() # could be super().__init__()

        self.hidden_size = hidden_layer_size # hidden state - standard thing

        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_layer_size) # embedding layer, input_size is the number of unique words, hidden_size is the size of the embedding

        self.gru = nn.GRU(input_size=hidden_layer_size, hidden_size=hidden_layer_size, batch_first=True) #Gated Recurrent Unit
        self.dropout = nn.Dropout(dropout_p) # dropout layer that helps to prevent overfitting
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden_state = self.gru(embedded)

        #hidden state - although we are not using it in this model, because the GRU takes care of it internally, other layers might use it
        return output, hidden_state 
    #-------------------------------------------------------------------------
#------------------------------------------------------------------------- 
#-------------------------------------------------------------------------
class DecoderRNN(nn.Module):
    #-------------------------------------------------------------------------
    def __init__(self, hidden_layer_size, output_size, device = None, max_length = 10, SOS_token = 0):
        super(DecoderRNN, self).__init__() # could be super().__init__()

        self.embedding = nn.Embedding(num_embeddings = output_size, embedding_dim = hidden_layer_size)

        self.gru = nn.GRU(input_size = hidden_layer_size, hidden_size = hidden_layer_size, batch_first = True)
        
        self.out = nn.Linear(in_features = hidden_layer_size, out_features = output_size)

        if not device: 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.max_length = max_length
        self.SOS_token = SOS_token

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, encoder_outputs, encoder_hidden, target_tensor = None):
        
        batch_size = encoder_outputs.size(0)

        # shape: (batch_size, 1) - fill with SOS_token
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class BahdanauAttention(nn.Module):
    #-------------------------------------------------------------------------
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__() #standard thing

        self.Wa = nn.Linear(hidden_size, hidden_size) # weight matrix for the query
        self.Ua = nn.Linear(hidden_size, hidden_size) # weight matrix for the keys
        self.Va = nn.Linear(hidden_size, 1) # weight matrix for the attention scores
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, query, keys):
        """
        Notes and summary:
        query - current decoder hidden state
        keys - encoder hidden states
        attention scores - indicate the relevance of each encoder hidden state to the current decoder 
            hidden state
        attention weights - These are the normalized attention scores, representing the probability 
            distribution over the encoder hidden states
        context vector - A weighted sum of the encoder hidden states, which is used to generate the 
            next decoder hidden state   
        """

        
        # attention scores
        attention_scores = self.Va(                      # apply another linear transformation to compute the attention scores
            torch.tanh(                        # tanh - standard thing
                self.Wa(query) + self.Ua(keys) # apply linear transformations to the query and keys an sum them
            )
        )

        # squeeze(2) - remove the dimension at idx 2, unsqueeze(1) - add a dimension at idx 1
        attention_scores = attention_scores.squeeze(2).unsqueeze(1) 

        attentin_weights = F.softmax(attention_scores, dim=-1) # apply softmax at the last dim to get the weights

        context_vector = torch.bmm(attentin_weights, keys) # bmm -> batch matrix-matrix product to product the context vector

        return context_vector, attentin_weights
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
##########################################################################
##########################################################################
##########################################################################
#######
#######
#######  Done until here
#######
#######
##########################################################################
##########################################################################
##########################################################################



#-------------------------------------------------------------------------
class AttnDecoderRNN(nn.Module):
    #-------------------------------------------------------------------------
    def __init__(self, hidden_size, output_size, device, sos_token, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()

        # 1 - embedding layer
        self.embedding = nn.Embedding(output_size, hidden_size) 

        # 2 - attention layer
        self.attention = BahdanauAttention(hidden_size) 

        # 3 - GRU layer. The GRU input size is 2 * hidden_size as it concatenates the embedded input 
        #   and the context vector from the attention mechanism. The output size is hidden_size
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True) 

        # 4 - linear layer that maps the GRU output to the desired output size (number of possible 
        #   output tokens).
        self.out = nn.Linear(hidden_size, output_size)


        self.dropout = nn.Dropout(dropout_p)

        # internal aux variables
        self.device = device
        self.SOS_token = sos_token
        self.max_length = max_length
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.sos_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def get_dataloader(batch_size,lang_prefixes, device, max_length = 10, EOS_token = 1):
    input_lang, output_lang, pairs = prepareData(
        lang1           = 'eng', 
        lang2           = 'fra', 
        lang1_prefixes  = lang_prefixes,
        reverse         = True
    )

    exit()
    n = len(pairs)
    input_ids = np.zeros((n, max_length), dtype=np.int32)
    target_ids = np.zeros((n, max_length), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part2(device, SOS_token, EOS_token, max_length, lang_prefixes):
    hidden_size = 128
    batch_size = 32


    input_lang, output_lang, train_dataloader = get_dataloader(
        batch_size      = batch_size,
        lang_prefixes   = lang_prefixes,
        device          = device,
        max_length      = max_length,
        EOS_token       = EOS_token
        )
    
    encoder = EncoderRNN(input_size = input_lang.n_words, hidden_layer_size=hidden_size).to(device)
    exit()
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
    
    train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)


    encoder.eval()
    decoder.eval()
    evaluateRandomly(encoder, decoder)    
#-------------------------------------------------------------------------
execute_part2(
    device          = result_part1['device'], 
    SOS_token       = result_part1['sos_token'], 
    EOS_token       = result_part1['eos_token'], 
    max_length      = result_part1['max_length'],
    lang_prefixes   = result_part1['lang_prefixes']
)


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

