#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import important packages

from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, BatchNormalization, GlobalAveragePooling1D, Masking
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import layers, callbacks
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from bs4 import BeautifulSoup
from pprint import pprint
import numpy as np
import pandas as pd
import csv
import re, string
from nltk import pos_tag
from nltk.corpus import stopwords
from pathlib import Path
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from glob import glob
import random
import os
import argparse
import time
import json
import h5py

np.random.seed(2021)