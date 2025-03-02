"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

print('\n\n')
print('-------------------------------------------------------------------------')
shard_size = int(100_000_000)

#------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"

dataset_path = "HuggingFaceFW/fineweb-edu"
dataset_split = "train"
#------------
# local_dir = "shakespeare_dataset"
# remote_name = None

# dataset_path = "notaphoenix/shakespeare_dataset"
# dataset_split = "training"

print(f'local_dir: {local_dir}')
print(f'remote_name: {remote_name}')
print(f'shard_size: {shard_size}')

print(f'    path: {dataset_path}\n    name: {remote_name}\n    split:{dataset_split}')


print('-------------------------------------------------------------------------')
print('init the tokenizer')
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token


print('-------------------------------------------------------------------------')
# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
print(f'Creating DATA_CACHE_DIR: {DATA_CACHE_DIR}')
os.makedirs(DATA_CACHE_DIR, exist_ok=True) # create DATA_CACHE_DIR, if it exists no error should be raised


print('-------------------------------------------------------------------------')
print('download the dataset')
fw = load_dataset(path = dataset_path, name=remote_name, split=dataset_split)





#-------------------------------------------------------------------------
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens

    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))

    tokens_np = np.array(tokens)

    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    
    tokens_np_uint16 = tokens_np.astype(np.uint16)

    return tokens_np_uint16
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)
#-------------------------------------------------------------------------



print('-------------------------------------------------------------------------')
cpu_count = os.cpu_count()
# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, cpu_count//2)


print(f'cpu_count: {cpu_count}')
print(f'nprocs: {nprocs}')

#-------------------------------------------------------------------------
with mp.Pool(nprocs) as pool: #processes pool
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)

    token_count = 0
    progress_bar = None

    #---------------------------------------------------
    # for tokens in pool.imap(func = tokenize, iterable = dataset_downloaded, chunksize = 16):
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder
    #---------------------------------------------------

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])


#-------------------------------------------------------------------------