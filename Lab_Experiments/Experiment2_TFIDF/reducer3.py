#!/usr/bin/env python3
"""Job 3 reducer: df, IDF and TF-IDF.
In : word \t doc,n,N   (sorted by word)
Out: word@doc \t tf=..,idf=..,tfidf=..
   tf    = n / N               (share of the document's words that are this word)
   idf   = log10(D / df)       (D = documents in corpus, df = documents containing the word)
   tfidf = tf * idf
D is passed in by the driver: -cmdenv TOTAL_DOCS=<D>
"""
import math
import os
import sys

D = int(os.environ["TOTAL_DOCS"])


def emit(word, postings):
    idf = math.log10(D / len(postings))
    for doc, n, total in postings:
        tf = n / total
        print(f"{word}@{doc}\ttf={tf:.4f},idf={idf:.4f},tfidf={tf * idf:.4f}")


current, postings = None, []
for line in sys.stdin:
    word, value = line.rstrip("\n").split("\t", 1)
    doc, n, total = value.split(",")
    if word != current:
        if current is not None:
            emit(current, postings)
        current, postings = word, []
    postings.append((doc, int(n), int(total)))
if current is not None:
    emit(current, postings)
