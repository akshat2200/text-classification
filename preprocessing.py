#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from imports import *
from load import dataframe_loader
from clean import *

def preprocess(text_column, label_column, max_nb_words, max_seq_len,
               test_size, oov_token = None):
    
    """
    preprocess(text, label, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, test_size = 0.25, oov_token = None):
    MAX_NB_WORDS: number of words to use, discarding the rest
    oov_token: out of vocabulary token
    
    """
    np.random.seed(100)
    
    # apply clean class on text
    clean = data_clean()
    text = text_column.apply(clean.denoice_text)
    
    tokenizer = Tokenizer(num_words = max_nb_words, oov_token = oov_token)
    tokenizer.fit_on_texts(text)
    sequence = tokenizer.texts_to_sequences(text)
    X, y = np.array(sequence), np.asarray(label_column).astype('float32').reshape((-1,1))
    word_index = tokenizer.word_index
    X = pad_sequences(X, maxlen = max_seq_len, padding = 'post', truncating = 'post')
    print('Found %s unique tokens.' % len(word_index))
    print('shape of text data', str(X.shape))
    
    # split data to training and holdout test set
    
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size = test_size,                                                               random_state = 1)
    
    # adding all to dictionary
    data = {}
    data["X_train"] = X_train
    data["X_holdout"] = X_holdout
    data["y_train"] = y_train
    data["y_holdout"] = y_holdout
    data["tokenizer"] = tokenizer
#     data["int2label"] = {0: label_1, 1: label_2}
#     data["label2int"] = {label_1: 0, label_2: 1}
    
    print("Number of Training data: ", str(data["X_train"].shape[0]))
    print("Number of Holdout data: ", str(data["X_holdout"].shape[0]))
    
    return data, word_index 


def word_embedding(word_index, embedding_size):
    
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_size))
    
    dir_path = f"./Data/glove.6B.{embedding_size}d.txt"
    with open(Path(dir_path), encoding = "utf8") as f:
        for line in tqdm(f, "Reading GloVe"):
            values = line.split()
            # extracting word as as first word in the line
            word = values[0].lower()
            if word in word_index:
                idx = word_index[word]
                # extract vectors as the remaining values in the line
                embedding_matrix[idx] = np.array(values[1:], dtype = "float32")
    f.close()
    print('Total %s word vectors.' % len(embedding_matrix))
    return embedding_matrix