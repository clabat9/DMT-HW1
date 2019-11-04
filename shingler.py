#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:55:47 2019

@author: ince
"""

import pandas as pd
import string
from collections import defaultdict
df = pd.read_csv('261K_lyrics_from_MetroLyrics.csv', usecols=['ID', 'lyrics'])

"""
For each lyric of a song, 
the set of shingles must be a set of natural numbers.
Before shingling a document, 
it is required to remove punctuations and convert all words in lower-case,  DONE
moreover, stopword removal, stemming and lemmatization are forbidden.  DONE
The length of each shingle must be 3.
You have to shingle only the lyric of the song. DONE
"""


def remove_punctuation_and_lower(s):
    """
    This will return a string without puntuation and all words are in lower-case.
    The punctuation symbols are get from the module 'string'.
    The set of punctuation symbols are '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    """
    return s.translate(str.maketrans('', '', string.punctuation)).lower()
    

###
### replace the column with the processed one  
###
df['lyrics'] = df['lyrics'].apply(remove_punctuation_and_lower)



def shingler(frame):
    shing_id = 0
    shingle_to_natural = dict()
    song_id_to_shingles = defaultdict(list)
    for idx, row in frame.iterrows():
        iterator = 0
        lyric_string_to_list = row['lyrics'].split(' ')
        while iterator+10 <= len(lyric_string_to_list):
            tmp_tuple = tuple(lyric_string_to_list[iterator:iterator+10])
            if tmp_tuple not in shingle_to_natural:
                shingle_to_natural[tmp_tuple] = shing_id
                shing_id += 1
            song_id_to_shingles[row['ID']].append(shingle_to_natural[tmp_tuple])
            iterator += 1
            
    with open('luci_blu_vuol_dire_solo_corri_k_10.tsv', 'w') as fptr:
        fptr.write("ID\tELEMENTS_IDS\n")
        for key in song_id_to_shingles:
            fptr.write("%s\t%s\n"%('id_'+str(key),str(song_id_to_shingles[key])))
    return "OK" #song_id_to_shingles, shingle_to_natural


a = shingler(df)