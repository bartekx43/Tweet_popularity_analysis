#Imports
import re
import nltk
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

#nltk.download('stopwords')
#nltk.download('punkt')

def get_data_from_file(filename):
  df = pd.read_csv(filename)

  start_tweet = 0
  tweets = df.content[start_tweet:]
  retweets = df.retweets[start_tweet:]

  return tweets, retweets


# Usuwanie zbędnych słów, linków, godzin, liczb
def clean_tweets(tweets):
  tweets = list(map(lambda s: re.sub(".*\.com.*","",s),tweets))
  tweets = list(map(lambda s: re.sub(".*http.*","",s),tweets))
  tweets = list(map(lambda s: re.sub(".*@.*","",s),tweets))
  tweets = list(map(lambda s: re.sub("[0-9]+","",s),tweets))
  return tweets


# Wykres popularności tweetów z czasem
def draw_chart(retweets):
  plt.plot(range(0,len(retweets)), retweets)
  plt.show()


def vectorize_tweets(tweets, retweets, n_occurences=50, own_stop_words=None):  
  # Tutaj dodatkowe słowa, które wykluczamy, oprócz zwykłych angielskich shitwords
  my_stop_words = text.ENGLISH_STOP_WORDS.union(own_stop_words)
  vectorizer = text.CountVectorizer(analyzer='word', min_df=n_occurences, stop_words=my_stop_words)
  X = vectorizer.fit_transform(tweets).toarray()
  # Normalizacja liczby retweetów
  r_max = max(retweets)
  Y = [r/r_max for r in retweets]
  
  return X, Y, vectorizer


def get_words(vectorizer):
  # Lista wszystkich użytych słów i jej rozmiar
  return vectorizer.get_feature_names_out()


# Zmiana reprezentacji z powrotem na słowa (tweet ofc nie brzmi dobrze, bo wielu słów brakuje)
def toWords(A):
  result = [all_words[i] for i,x in enumerate(A) if x != 0]
  return " ".join(list(result))

def get_linear_regression_model(X_train, y_train):
  model = LinearRegression()
  model = model.fit(X=X_train, y=y_train)
  return model

def check_accuracy(y_test, y_pred):
  err = [abs(y1-y2) for y1,y2 in zip(y_test, y_pred)]

  shuffled = y_pred.copy()
  random.shuffle(shuffled)
  err = [abs(y1-y2) for y1,y2 in zip(y_test, shuffled)]

  with open("result.txt", "a") as f:
    f.write(f"Error: {sum(err)/len(err)}\n")
    f.write(f"Random error: {sum(err)/len(err)}\n")


def get_shap(model, all_words):    
  words = []
  for i in range(len(all_words)):
    word = [0 for _ in range(len(all_words))]
    word[i] = 1
    words.append(word)

  words_weights = model.predict(words)
  words_zipped = [(x, y) for y, x in reversed(sorted(zip(words_weights, all_words)))]
  
  with open("result.txt", "a") as f:
    f.write(f"SHAP: {words_zipped[:5]}\n")
  
  return words_zipped

#Pomysł na sprawdzenie SHAP: róznica zbiorów najczestszych słow popuarnych z tweetow i niepopularnych tweetów

def get_mlp_regression_model(X_train, y_train, epochs, layers):
  model = MLPRegressor(random_state=1, max_iter=epochs, hidden_layer_sizes=layers).fit(X_train, y_train)
  return model


if(__name__ == "__main__"):

  ns = [500, 300, 200, 150, 100, 80, 60, 50, 40, 30, 20, 10]

  for n in ns: 
    # Przygotowanie danych
    tweets, retweets = get_data_from_file('trumptweets.csv')
    tweets = clean_tweets(tweets)
    draw_chart(retweets)
    X, Y, vectorizer = vectorize_tweets(tweets, retweets, n, ["twitter", "realdonaldtrump", "nd", "th"])
    all_words = get_words(vectorizer)
    print(f"Liczba uzytych slow kluczowych: {len(all_words)}")
    #print(all_words)

    for train_share in [0.9, 0.8, 0.7]:
      X_train, X_test, y_train, y_test  = train_test_split(
              X, 
              Y,
              train_size=train_share, 
              random_state=1234,
              shuffle=True)
      
      with open("result.txt", "a") as f:
        f.write(f"LINEAR REG words_n={n} train_share={train_share}\n")


      model = get_linear_regression_model(X_train, y_train)
      y_pred = model.predict(X_test)
      check_accuracy(y_test, y_pred)
      shap = get_shap(model, all_words)

      #print(shap[:10])
      for epochs in [50, 100, 200, 500, 1000]:
        for layers in [(64,), (128,), (256,), (512,), (128,32), (128, 64), (256, 64), (512, 128), (512, 64), (128, 64, 32), (64, 32, 8)]:

          with open("result.txt", "a") as f:
            f.write(f"MLP REG words_n={n} train_share={train_share} epochs={epochs} layers={layers}\n")

          model = get_mlp_regression_model(X_train, y_train, epochs, layers)
          y_pred = model.predict(X_test)
          check_accuracy(y_test, y_pred)
      
      #print(get_shap(model, all_words)[:10])
