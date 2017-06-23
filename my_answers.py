import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    i = 0
    while i + window_size < len(series):
        X.append(series[i:(i + window_size)])
        i += 1
        
    y = series[window_size:]
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:window_size])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    model = Sequential()
    model.add(LSTM(32, input_shape=(7, 1)))
    model.add(Dense(1))
    return model


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    unique_chars = set(text)
    print(unique_chars)
    
    # remove as many non-english characters and character sequences as you can
    # instead of removing the non-english characters, use regex to keep only the
    # authorized characters. According the the evalution rubric, these authorized characters are:
    # (English characters should include string.ascii_lowercase and 
    #  punctuation includes [' ', '!', ',', '.', ':', ';', '?'] 
    #  (space, eclamation mark, comma, period, colon, semicolon, question mark))
    import re
    text = re.sub((r'[^a-z!,.:;?]'), ' ', text)
    text = text.replace('  ',' ')
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    i = 0
    
    while i + window_size < len(text):
        inputs.append(text[i:i + window_size])
        outputs.append(text[i + window_size])
        i += step_size

    return inputs,outputs
