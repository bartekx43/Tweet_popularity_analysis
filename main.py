# -*- coding: utf-8 -*-

# Imports
import re
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import pickle
from wordcloud import WordCloud


# Pobranie tweetów z pliku
def get_data_from_file(filename):
    df = pd.read_csv(filename)
    start_tweet = 0
    tweets = df.content[start_tweet:]
    retweets = df.retweets[start_tweet:]

    return tweets, retweets


# Usuwanie zbędnych słów, linków, godzin, liczb
def clean_tweets(tweets):
    tweets = list(map(lambda s: re.sub(".*\.com.*", "", s), tweets))
    tweets = list(map(lambda s: re.sub(".*http.*", "", s), tweets))
    tweets = list(map(lambda s: re.sub(".*@.*", "", s), tweets))
    tweets = list(map(lambda s: re.sub("[0-9]+", "", s), tweets))

    return tweets


# Wykres popularności kolejnych tweetów
def draw_chart(retweets):
    plt.plot(range(0, len(retweets)), retweets)
    plt.title("Liczba retweetów")
    plt.show()


def vectorize_tweets(tweets, retweets, n_occurences=50, own_stop_words=[]):
    # Dodatkowe słowa do wykluczenia oprócz defaultowych
    my_stop_words = text.ENGLISH_STOP_WORDS.union(own_stop_words)
    vectorizer = text.CountVectorizer(analyzer='word', min_df=n_occurences, stop_words=my_stop_words)
    X = vectorizer.fit_transform(tweets).toarray()
    # Normalizacja liczby retweetów
    r_max = max(retweets)
    Y = [r / r_max for r in retweets]

    return X, Y, vectorizer


def get_words(vectorizer):
    # Lista wszystkich użytych słów
    return vectorizer.get_feature_names_out()


def get_linear_regression_model(X_train, y_train):
    return LinearRegression().fit(X=X_train, y=y_train)


def get_mlp_regression_model(X_train, y_train, epochs, layers):
    return MLPRegressor(random_state=1, max_iter=epochs, hidden_layer_sizes=layers).fit(X_train, y_train)


def check_improvement(y_test, y_pred):
    err = [abs(y1 - y2) for y1, y2 in zip(y_test, y_pred)]

    shuffled = y_pred.copy()
    random.shuffle(shuffled)
    err_random = [abs(y1 - y2) for y1, y2 in zip(y_test, shuffled)]
    print(f"Random error: {sum(err_random)/len(err_random)}")
    print(f"Obtained error: {sum(err)/len(err)}")


# Analiza wpływu poszczególnych słów na wartość predykcji
def get_shap(model, all_words):
    words = []
    for i in range(len(all_words)):
        word = [0 for _ in range(len(all_words))]
        word[i] = 1
        words.append(word)

    words_weights = model.predict(words)
    words_zipped = [(x, y) for y, x in reversed(sorted(zip(words_weights, all_words)))]

    return words_zipped


# Rysowanie chmury słów o największym wpływie na popularność
def draw_cloud(shap):
    words_dict = {s[0]: s[1] for s in shap}

    wc = WordCloud(max_words=200)
    wc.fit_words(words_dict)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def predict_own(tweets, all_words, model):
    X = [[1 if x in t else 0 for x in all_words] for t in tweets]
    print(f"Standardized popularity: {model.predict(X)}")


def save_model(model, filename):
    filename = filename
    pickle.dump(model, open(filename, 'wb'))


def load_model(filename):
    return pickle.load(open(filename, 'rb'))


if __name__ == "__main__":
    # Pobranie tweetów z pliku
    tweets, retweets = get_data_from_file('trumptweets.csv')
    tweets = clean_tweets(tweets)

    draw_chart(retweets)

    # LinearRegression
    X_reg, Y_reg, vectorizer_reg = vectorize_tweets(tweets, retweets, 50, ["twitter", "realdonaldtrump", "nd", "th"])
    all_words_reg = get_words(vectorizer_reg)

    X_train, X_test, y_train, y_test = train_test_split(
        X_reg,
        Y_reg,
        train_size=0.7,
        random_state=1234,
        shuffle=True)

    # model_reg = get_linear_regression_model(X_train, y_train)
    # save_model(model_reg, "model_reg.sav")

    model_reg = load_model("model_reg.sav")

    y_pred = model_reg.predict(X_test)
    check_improvement(y_test, y_pred)
    shap = get_shap(model_reg, all_words_reg)
    draw_cloud(shap)

    # MLPRegression
    epochs = 25
    layers = (512,)
    X_mlp, Y_mlp, vectorizer_mlp = vectorize_tweets(tweets, retweets, 10, ["twitter", "realdonaldtrump", "nd", "th"])
    all_words_mlp = get_words(vectorizer_mlp)

    X_train, X_test, y_train, y_test = train_test_split(
        X_mlp,
        Y_mlp,
        train_size=0.7,
        random_state=1234,
        shuffle=True)

    # model_mlp = get_mlp_regression_model(X_train, y_train, epochs, layers)
    # save_model(model_mlp, "model_mlp.sav")

    model_mlp = load_model("model_mlp.sav")

    y_pred = model_mlp.predict(X_test)
    check_improvement(y_test, y_pred)
    shap = get_shap(model_mlp, all_words_mlp)
    draw_cloud(shap)

    # Własne predykcje
    sample_tweets = ["I would like that the whole world be in peace and all people love one another",
         "We will build a great wall and mexico will pay for it!",
         "I don't like eating banana and orange because it makes me sick and I need to go to the doctor",
         "president obama is sharing fake news!",
         "We should immediately ban immigrants from illegally coming to our country!"]

    predict_own(sample_tweets, all_words_reg, model_reg)
    predict_own(sample_tweets, all_words_mlp, model_mlp)
