
import pywt
import h5py
import wrcoef
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from statistics import mean, median, mode, stdev
import csv
import h5py
import pywt
import eeglib
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku 
from tensorflow.keras.models import load_model
from tensorflow import keras
from gtts import gTTS
from IPython.display import Audio
 
def dataCollect(fileName):
  with h5py.File(fileName, 'r') as hf:
    ls = list(hf.keys())
    data_raw = hf.get(ls[0])
    data_raw = np.array(data_raw)
    new_data = []
    for element in data_raw:
      new_data.extend(element)
  new_data = (((np.array(new_data) / 2**10) - 0.5) * 3e6) / 41780
  new_data = new_data.tolist()
  return new_data

def dataCollectFiltered(fileName):
  with h5py.File(fileName, 'r') as hf:
    ls = list(hf.keys())
    data_raw = hf.get(ls[0])
    data_raw = np.array(data_raw)
  data = data_raw.tolist()
  return data

def dataCollectFitur(fileName):
  with h5py.File(fileName, 'r') as hf:
    ls = list(hf.keys())
    data_raw = hf.get(ls[0])
    data_raw = np.array(data_raw)
  data = data_raw.tolist()
  return data

def filterButter(Fs=1000, fp=np.array([8, 30]), fs=np.array([1, 35]), Ap=0.4, As=3):
  wp = fp/(Fs/2)
  ws = fs/(Fs/2)
  N, wc = signal.buttord(wp, ws, Ap, As, analog=True)
  print('Order of the filter=', N)
  print('Cut-off frequency=', wc)
  z, p = signal.butter(N, wc, 'bandpass')
  print('Numerator Coefficients:', z)
  print('Denominator Coefficients:', p)
  wz, hz = signal.freqz(z, p)
  return z, p

def fiturEkstraksi(data):
  dfa = eeglib.features.DFA(data)
  hfd = eeglib.features.HFD(data)
  lzc = eeglib.features.LZC(data)
  pfd = eeglib.features.PFD(data)
  hActivity = eeglib.features.hjorthActivity(data)
  hComplexity = eeglib.features.hjorthComplexity(data)
  hMobility = eeglib.features.hjorthMobility(data)
  samp = eeglib.features.sampEn(data)
  return dfa, hfd, lzc, pfd, hActivity, hComplexity, hMobility,samp

def predictANN(dfa, hfd, lzc, pfd, hActivity, hComplexity, hMobility,samp):
  model_ANN = load_model('model_ANN.h5')
  score = [dfa, hfd, lzc, pfd, hActivity, hComplexity, hMobility, samp]
  score = np.array([score])
  y_score = model_ANN.predict_classes(score)
  if y_score == np.array([0]):
    hasil = 'a'
  elif y_score == np.array([2]):
    hasil = 'ku'
  elif y_score == np.array([3]):
    hasil = 'ma'
  elif y_score == np.array([1]):
    hasil = 'kan'
  return hasil

def prediksiRNN(seed_text, next_words = 1):
  model = load_model('model_RNN.h5')
  tokenizer = Tokenizer()
  data = open('Prediksi.TXT').read()
  corpus = data.lower().split("\n")
  tokenizer.fit_on_texts(corpus)
  total_words = len(tokenizer.word_index) + 1
  input_sequences = []
  for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
      n_gram_sequence = token_list[:i+1]
      input_sequences.append(n_gram_sequence)
  # pad sequences 
  max_sequence_len = max([len(x) for x in input_sequences])
  input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
  for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
      if index == predicted:
        output_word = word
        break
    seed_text += " " + output_word
  print(seed_text)
  return seed_text

def text_to_speech(word):
  tts = gTTS(word)
  tts.save('1.wav')
  sound_file= '1.wav'
  Audio(sound_file, autoplay= True)