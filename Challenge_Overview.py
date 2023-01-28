from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
import nltk
import csv


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


corpus = []

with open("DETAILS.csv", "r", encoding="UTF-8-sig") as DETAILS_csv:

    reader = csv.reader(DETAILS_csv)

    for item in reader:
        if item[0] == "id":
            continue
        else:
            corpus.append([item[0], item[1]])

stop_words = ""

for stop_word in stopwords.words("english"):
    stop_words += stop_word
    stop_words += " "

stop_words = list(set(word_tokenize(stop_words)))

with open("Challenge_Overview.csv", "w", newline="", encoding="utf-8-sig") as AAA_csv:

    writer = csv.writer(AAA_csv)

    fileHeader = ["id", "Challenge_Overview"]
    writer.writerow(fileHeader)

    for detail in corpus:

        words = word_tokenize(detail[1].lower())
        words = [word for word in words if word.isalnum()]
        words = [word for word in words if word.isalpha()]
        words = [word for word in words if word not in stop_words]
        # words = [word for word in words if word not in list(punctuation)]

        words_net = nltk.pos_tag(words)

        lemmatizer = WordNetLemmatizer()

        detail[1] = ""

        for word_net in words_net:
            word = lemmatizer.lemmatize(word=word_net[0], pos=get_wordnet_pos(word_net[1]))
            detail[1] += word
            detail[1] += "\n"

        writer.writerow([detail[0], detail[1]])
        