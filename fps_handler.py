#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:13:27 2019

@author: ince
"""

import pandas as pd
import string



def remove_punctuation_and_lower(s):
    """
    This will return a string without puntuation and all words are in lower-case.
    The punctuation symbols are get from the module 'string'.
    The set of punctuation symbols are '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    """
    return s.translate(str.maketrans('', '', string.punctuation)).lower()


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

test_uno = pd.read_csv('test_uno.tsv', sep = '\t', usecols=['Estim_Jacc','idx1', 'idx2'])


###
### We're interested only in the id not not in the entire string, 
### so we replace the entries from id_12345 to 12345.
###

test_uno['idx1'] = test_uno['idx1'].apply(lambda x: x.split('_')[1])
test_uno['idx2'] = test_uno['idx2'].apply(lambda x: x.split('_')[1])

print("After the execution of the Near Duplicates tool, we've found:",len(test_uno), "candidates.")

###
###
###

lyrics = pd.read_csv('~/Desktop/uni/1y2s/dmt/HW1/part_2/dataset/261K_lyrics_from_MetroLyrics.csv', usecols= ['ID', 'lyrics'])

print("\n",lyrics.head())



    

###
### In order to perform the real jaccard similarity we have to take the lyrics set
### in the same situation it was before the shingling. So we removed the punctuation and all the upper cases.
###
lyrics['lyrics'] = lyrics['lyrics'].apply(remove_punctuation_and_lower)
print("After processing the lyrics set, the result looks like this")
print("\n",lyrics.head())

###
### Perform the TRUE jaccard similarities on all 42000 pairs
###
similarities = []
for idx,row in test_uno.iterrows():
    A = list(lyrics.loc[lyrics['ID'] == int(row[1])]['lyrics'])[0].split(' ')
    B = list(lyrics.loc[lyrics['ID'] == int(row[2])]['lyrics'])[0].split(' ')
    #test_uno['peppino'].loc[idx] = jaccard_similarity(A,B)
    similarities.append(jaccard_similarity(A,B))

test_uno['True_Jacc'] = pd.Series(similarities)

del similarities


###
### These are the 'statistical' false positive candidates. 
### The pairs for which, if i build this wald test: H0: J_t >= 0.88 ; H1: J_t < 0.88
### You refuse the null hypothesis, and as explained in the report, we're going to check only  these pairs.
###
print("\n",test_uno.loc[test_uno['Estim_Jacc']-2*(test_uno['Estim_Jacc'].std())< .88])

#test_uno.loc[test_uno['Estim_Jacc']-2*(test_uno['Estim_Jacc']*(1-(test_uno['Estim_Jacc'])))**0.5 < .88].loc[(test_uno['True_Jacc']<.88) & (test_uno['True_Jacc'] != 0)]


print("\n",len(test_uno.loc[test_uno['Estim_Jacc']-2*(test_uno['Estim_Jacc'].std())< .88].loc[(test_uno['True_Jacc']<.88) & (test_uno['True_Jacc'] != 0)]))
print("\n As you can see the length of pairs to check is decreased by a factor 7 and the number of false positive still the same")