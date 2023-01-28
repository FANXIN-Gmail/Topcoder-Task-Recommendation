from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import nltk

# ISO-8859-1
# utf-8
# gbk


def get_wordnet_pos(tag):

    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


FIRST_2_FINISH = pd.read_csv("FIRST_2_FINISH.csv", encoding="ISO-8859-1")
name = FIRST_2_FINISH["name"]
result = []

stop_words = ""

for stop_word in stopwords.words("english"):
    stop_words += stop_word
    stop_words += " "

stop_words = list(set(word_tokenize(stop_words)))

for value in name.values:

    words = word_tokenize(value.lower())
    words = [word for word in words if word.isalnum()]
    words = [word for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]

    words_net = nltk.pos_tag(words)

    lemmatizer = WordNetLemmatizer()

    value = ""

    for word_net in words_net:
        word = lemmatizer.lemmatize(word=word_net[0], pos=get_wordnet_pos(word_net[1]))
        value += word
        value += "\n"

    result.append(value)



