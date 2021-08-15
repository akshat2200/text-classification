#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from imports import *
from load import dataframe_loader
from clean import *
from preprocessing import *
from model import rnn_model

def train(x_train, y_train, word_index, x_holdout, y_holdout, save_path):
    
    print("\nTraining...")
    model = rnn_model(word_index, args.units, args.n_layers, args.cell, args.bidirectional,
                     args.embedding_size, args.max_seq_len, args.dropout, args.loss, args.optimizer,
                     args.output_length, args.activation, args.metrics)
    model.summary()
    plot_model(model, show_shapes = True, to_file = os.path.join(args.results_dir, timestamp, "model.pdf"))
    
    def optimal_model_name(data_name):
        # define model name
        data_name = args.file_name
        RNN_CELL = args.cell
        SEQUENCE_LENGTH = args.max_seq_len
        EMBEDDING_SIZE = args.embedding_size
        N_WORDS = args.max_nb_words
        N_LAYERS = args.n_layers
        UNITS = args.units
        OPTIMIZER = args.optimizer
        BATCH_SIZE = args.batch_size
        DROPOUT = args.dropout
        
        name_model = f"{data_name}-{RNN_CELL}-seq-{SEQUENCE_LENGTH}-em-{EMBEDDING_SIZE}-w-{N_WORDS}-layers-{N_LAYERS}-units-{UNITS}-opt-{OPTIMIZER}-BS-{BATCH_SIZE}-d-{DROPOUT}"
        if args.bidirectional:
            # adding 'bid' for bidirectional
            name_model = "bid-" + name_model
        return name_model
    
    model_name = optimal_model_name(args.file_name)

    tensorboard = TensorBoard(log_dir = os.path.join(args.results_dir, timestamp,'log/'), 
                              histogram_freq = 0.1, write_graph = True, write_grads = True,
                             write_images = True, embeddings_freq = 0.5, update_freq = 'batch')
    
    early_stopping = callbacks.EarlyStopping(
        		min_delta = 0.001, # minimium amount of change to count as an improvement
        		patience= args.patience, # how many epochs to wait before stopping
        		restore_best_weights = True,)
    
    history = model.fit(x_train, y_train,
                        batch_size = args.batch_size,
                        epochs = args.epochs,
                        validation_data = (x_holdout, y_holdout),
                        callbacks = [early_stopping, tensorboard],
                        verbose = 1)
    
    print("\nSaving model with best weights...")
    tf.keras.models.save_model(model, save_path)
    pprint(history.history)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "This is RNN classification project.")
    parser.add_argument('-f', '--file_name', default = 'train.csv', type = str, help = 'This is training file name with .csv extension.(default=train.csv)')
    parser.add_argument('-tc', '--text_column', default = 'text', type = str, help = 'Text column to train model.(default=text)')
    parser.add_argument('-lc', '--label_column', default = 'target', type = str, help = 'Target or label column for text data.(default=target)')
    parser.add_argument('-n', '--max_nb_words', default = 20000, type = int, help = 'Maximum vocabulary size of training data, rest will be discarted.(default=20000)')
    parser.add_argument('-s', '--max_seq_len', default = 300, type = int, help = "Maximum length of words in each text.(default=300)")
    parser.add_argument('-t', '--test_size', default = 0.25, type = float, help = 'The fraction of validation data.(default=0.25)')
    parser.add_argument('-u', '--units', default = 300, type = int, help = 'Number of units for RNN cell.(default=300)')
    parser.add_argument('-l', '--n_layers', default = 1, type = int, help = 'Number of hidden layers.(default=1)')
    parser.add_argument('-c', '--cell', default = LSTM, type = object, help = 'Type of RNN cell.(default=LSTM)')
    parser.add_argument('-b', '--bidirectional', default = False, type = bool, help = 'Whether RNN cell is BiDirectional.(delafult=False)')
    parser.add_argument('-e', '--embedding_size', default = 300, type = int, help = 'Pre trained GloVe 6B vocabulary embeddings.(default=300)')
    parser.add_argument('-d', '---dropout', default = 0.5, type = float, help = 'Dropout rate in sigmoid/softmax layer.(default=0.5)')
    parser.add_argument('-lo', '--loss', default = 'binary_crossentropy', type = str, help = 'Loss function for RNN.(default=binary_crossentropy)')
    parser.add_argument('-o', '--optimizer', default = 'adam', type = str, help = 'Optimizer function for RNN.(default=adam)')
    parser.add_argument('-cl', '--output_length', default = 1, type = int, help = 'Number of target classes.(default=1)')
    parser.add_argument('-a', '--activation', default = 'sigmoid', type = str, help = 'Activation function for output layer.(default=sigmoid)')
    parser.add_argument('-m', '--metrics', default = 'binary_accuracy', type = str, help = 'Measure for optimization of model.(default=binary_accuracy)')
    parser.add_argument('-p', '--patience', default = 20, type = int, help = 'Early stopping before waiting for additional epochs.(default=20)')
    parser.add_argument('-bs', '--batch_size', default = 64, type = int, help = 'Mini-Batch size.(default=64)')
    parser.add_argument('--epochs', default = 500, type = int, help = 'Number of epochs.(default=500)')
    parser.add_argument('--results_dir', default = './Results/', type = str, help =  'The results folder includes log, model, vocabulary and some images.(default=./Results/)')
    args = parser.parse_args()
    print('Parameters:', args, '\n')
    
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    timestamp = time.strftime("%d-%m-%Y-%H-%M", time.localtime(time.time()))
    os.mkdir(os.path.join(args.results_dir, timestamp))
    os.mkdir(os.path.join(args.results_dir, timestamp, 'log/'))
    
    dff = dataframe_loader(file_name = args.file_name)
    data, word_index = preprocess(text_column = dff[args.text_column],
                                  label_column = dff[args.label_column],
                                  max_nb_words = args.max_nb_words,
                                  max_seq_len = args.max_seq_len,
                                  test_size = args.test_size)

    with open(os.path.join(args.results_dir, 'vocab_file.json'), "w") as outfile:
         json.dump(word_index, outfile)
    
    train(data["X_train"], data["y_train"], word_index, data["X_holdout"], data["y_holdout"], os.path.join(args.results_dir, timestamp, 'Bi_LSTM.h5'))