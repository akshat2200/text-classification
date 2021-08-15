#!/usr/bin/env python
# coding: utf-8

# In[1]:


from imports import *
from load import dataframe_loader
from clean import *
from preprocessing import *
from model import rnn_model

def test(model, x_test):
    print('Running model on test data')
    y_pred_one_hot = model.predict(x = x_test, batch_size = args.batch_size, verbose = 1)
    if(args.multiple):
        y_pred = tf.math.argmax(y_pred_one_hot, axis = 1)
    else:
        y_pred = y_pred_one_hot
    
    return y_pred
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is the Bi-LSTM test project.')
    parser.add_argument('-f', '--file_name', default = 'test.csv', type = str, help = 'This is test file name with .csv extension.(default=test.csv)')
    parser.add_argument('-tc', '--text_column', default = 'text', type = str, help = 'Text column to test model.(default=text)')
    parser.add_argument('--results_dir', type = str, help='The results dir including log, model, vocabulary and some images.')
    parser.add_argument('-bs', '--batch_size', default = 64, type = int, help = 'Mini-Batch size.(default=64)')
    parser.add_argument('-s', '--max_seq_len', default = 300, type = int, help = "Maximum length of words in each text.(default=300)")
    parser.add_argument('-m', '--multiple', default = False, type = bool, help = "Whether multi-classification or not.(default=False)")  
    args = parser.parse_args()
    
    # load test data
    dff_test = dataframe_loader(file_name = args.file_name)
    print(f"Shape of test data: {dff_test.shape}")
    
    # apply clean class on text
    clean = data_clean()
    text = dff_test[args.text_column].apply(clean.denoice_text)
    
    # load vocabulary from training
    word_dict = json.load(open(os.path.join("./Results/", "vocab_file.json"), 'r'))
    vocabulary = word_dict.keys()
    
    X_test = [[word_dict[each_word] if each_word in vocabulary else 1 for each_word in each_sentence.split()] for each_sentence in text]
    X_test = pad_sequences(X_test, maxlen = args.max_seq_len,
                            padding='post', truncating='post')
    print("Shape of test data: {}\n".format(np.shape(X_test)))
    
    print("\nLoading model...")
    model = tf.keras.models.load_model(Path(args.results_dir))
    y_pred = test(model, x_test = X_test)
    
    # adding predictions back to test data
    dff_test['y_pred'] = y_pred
    print(f"Shape of output data: {dff_test.shape}")
    
    # write to csv file
    print('Writting predictions back to test data file')
    dff_test.to_csv(os.path.join('./Data/output.csv'))