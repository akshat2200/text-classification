#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from imports import *
from load import dataframe_loader
from clean import *
from preprocessing import *


def rnn_model(word_index,
              units,
              n_layers,
              cell,
              bidirectional,
              embedding_size,
              sequence_length,
              dropout,
              loss,
              optimizer,
              output_length,
              activation,
              metrics):
    
    """
    rnn_model(word_index, units = 128, n_layers = 1, cell = LSTM, bidirectional = False, embedding_size = 100, sequence_length = 100, droupout = 0.3, loss = "categorical_crossentropy", optimizer = "adma", output_length = 2):
    word_index: 
    units: number of units (RNN_CELL ,nodes) in each layer
    n_layers: number of CELL layers
    cell: the RNN cell to use, LSTM in this case
    bidirectional: whether it's a bidirectional RNN
    embedding_size: N-Dimensional GloVe embedding vectors
    sequence_length: max number of words in each sentence
    dropout: dropout rate
    embedding_matrix: the embedding matrix
    output_length: number of classes
    """
    
    embedding_matrix = word_embedding(word_index, embedding_size)
    model = Sequential()
    
    # add embedding layer
    model.add(Embedding(len(word_index) + 1,
                       embedding_size,
                       input_length = sequence_length,
                       weights=[embedding_matrix],
                       trainable = False,
                       mask_zero = True
                      ))
    
    # loop for number of layers
    for i in range(n_layers):
        if i == n_layers - 1:
            # for last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences = False,                                              recurrent_dropout = 0.2)))
            else:
                model.add(cell(units, return_sequences = False, recurrent_dropout = 0.2))
        else:
            # for hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences = True,                                              recurrent_dropout = 0.2)))
            else:
                model.add(cell(units, return_sequences = True, recurrent_dropout = 0.2))
    
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(units/2, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(output_length, activation = activation))
    
    # compile the model
    model.compile(optimizer = optimizer, loss = loss, metrics = [metrics])
    
    return model