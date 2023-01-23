#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:53:37 2022

@author: sebastian
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import scipy as sp
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import pickle
import numpy as np
import seaborn as sb
import matplotlib as plt
from loaders import Boundary_Loader, Output_Loader, Word_Loader

from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def get_apiKey():
    home = os.path.expanduser("~")   
    fp = open(home + '/' + 'openai.txt',
               encoding='utf-8-sig')
    return fp.read().replace('\n', '')
def get_plain_text(story): 
    fp = open("../" + story + "/" + story+ "Clean.txt", encoding='utf-8-sig')
    data = fp.read()
    # there shouldn't be any \\ in the text anymore
    text_plain = data.replace('\\"', '')
    text_plain = text_plain.replace('...', '.')
    return text_plain
def get_sentence_bounds(text):
    sentences = sent_tokenize(text)
    index = 0
    onset_list = []
    offset_list = []
    wordsum = 0;
    word_ends = []
    word_start = []
    # loop through all the sentences
    for s in (sentences):
        if story == "Tunnel": # fix annoying compound nouns in Tunnel
            s = s.replace("-", "")
        # get the word_list for the sentence:
        tmp_words = word_tokenize(re.sub(r"[^a-zA-Z0-9 ]", "", ''.join(s)))
        # fix wanna gonna ----
        for idx, w in enumerate(tmp_words):
            if w.lower() == "wan":
                tmp_words[idx] = "wanna"
            elif w.lower() == "gon": 
                tmp_words[idx] = "gonna"
        while 'na' in tmp_words: 
            tmp_words.remove('na')
        # ------------
        # add words to the sum
        wordsum = wordsum + len(tmp_words)
        #DeBUGGING w_index = s.casefold().find(words[index].casefold())
        # add the indexed word's onset to the list
        onset_list.append(w_onsets[index])
        
        # FIX NAN problems in Tunnel!If there is a nan, we use the the next onset
        if story == "Tunnel":
            tmpwin = 0
            while np.isnan(onset_list[-1]):
                tmpwin = tmpwin + 1
                onset_list[-1] = w_onsets[index+tmpwin]
        word_start.append(words[index])
        #s = s[index+len(words[index])+1:]
        index = index + 1 # move the word index forward
        l_index = 0
        # as long as we are in this sentence (<wordsum) and the word can be found
        while l_index >= 0 and index < wordsum:
            # extra check to make sure we don't exceed the word-list
            if index < len(words)-1:
                # find the last occurence of this word in the sentence
                l_index = s.casefold().rfind(words[index].casefold())
                # if the word is not found, try to fix it!
                # this may be because of im, ive and id
                if l_index == -1:
                    l_index = s.casefold().replace("'", "") .rfind(words[index].casefold())
                # could be nan...
                if l_index == -1:
                    print(index)
            # keep increasing the index
            index = index + 1

         #   print(words[index])
       # index = index -1
        word_ends.append(words[index-1])
        offset_list.append(w_offsets[index-1])
        # FIX NAN problems in Tunnel!
        if story == "Tunnel":
            tmpwin = 0
            # if there is a nan in Tunnel, we appen the previous offset
            while np.isnan(offset_list[-1]):
                tmpwin = tmpwin + 1
                offset_list[-1] = w_offsets[index-1-tmpwin]
            
        wordsum = index
    # we should have arrived at the end
    sentence_bounds = (np.array(onset_list[1:]) + np.array(offset_list[:-1]))/2
    assert(all(offset_list[:-1]< onset_list[1:]))
    return sentence_bounds, sentences, onset_list, offset_list, word_start, word_ends

def get_p_valueShuffle(bounds1, bounds2, n_rand = 1000):
    distance_real = sp.spatial.distance.hamming(bounds1, bounds2, w=None)
    rand_dists = np.ones(n_rand,) * np.inf
    for rr in range(n_rand):
        vec2 = bounds2.copy()
        np.random.shuffle(vec2)
        rand_dists[rr] = sp.spatial.distance.hamming(
            bounds1, vec2, w=None)
    p_val = np.sum(rand_dists <= distance_real)/n_rand
    return p_val, rand_dists

def get_p_valueCircShuffle(bounds1, bounds2, n_rand = 1000):
    distance_real = sp.spatial.distance.hamming(bounds1, bounds2, w=None)
    rand_dists = np.ones(n_rand,) * np.inf
    for rr in range(n_rand):
        vec2 = bounds2.copy()
        vec2 = np.roll(vec2, np.random.randint(0,len(vec2)))
        rand_dists[rr] = sp.spatial.distance.hamming(
            bounds1, vec2, w=None)
    p_val = np.sum(rand_dists <= distance_real)/n_rand
    return p_val, rand_dists
def get_human_boundary_vector(sentence_bounds, bounds_s):
    boundary_vector = np.zeros(len(sentences)-1,)
    for bound in bounds_s:
        boundary_vector[np.absolute(sentence_bounds - bound).argmin()]  = True
    return boundary_vector
    
#%% versions and parameters
n_iter = 6
#%%
story_index = 2  # Monkey', 'Tunnel', 'Pieman'
version_index =  0 # 'long ' ,''
vecs = []
rand_function = get_p_valueShuffle
for it in range(n_iter):
   
    
    stories = ['Monkey', 'Tunnel', 'Pieman']
    
    versions = ['long ' ,'']
    
    story = stories[story_index]   
    version = versions[version_index] #"long " # "long " (include the space!)
    
    
    text_plain = get_plain_text(story)
    
    bl = Boundary_Loader(story)
    ol = Output_Loader(story,version, n_iter)
    wl = Word_Loader(story, text_plain)
    
    boundary_vector_gpt = ol.load_bound_vec(it)
    events = ol.load_event_list(it)
    responses = ol.load_response_list(it)
    vecs.append(boundary_vector_gpt)
    words, w_onsets, w_offsets = wl.load_words_and_times()
    #% get the human boundaries too
    
    # manoj's bounds
    Bounds_manoj = bl.get_Manoj_filtered_bounds_ms()
    Bounds_all1, Bounds_all2  = bl.load_human_bounds_ms()
    
    # get the sentence bounds
    sentence_bounds, sentences, onset_list, offset_list, starting_words, ending_words = (
        get_sentence_bounds(text_plain))
    # % get the sentence boundaries for the story
    
    Bounds_ms = Bounds_all1
    
    if story == "Pieman":
        print("warning: check which boundaries are used")
        Bounds_ms = Bounds_all2
    #% compare to the boundaries from the first run
    boundary_vector = get_human_boundary_vector(sentence_bounds, Bounds_ms/1000)
  
    
    distance_real = sp.spatial.distance.hamming(boundary_vector, boundary_vector_gpt, w=None)
    
    # p_val, rand_dists = get_p_valueShuffle(boundary_vector, boundary_vector_gpt, 100000)
    # print('p value = ' + str(p_val))
    
    p_val, rand_dists = rand_function(boundary_vector, boundary_vector_gpt, 100000)
    print('hamming distance = ' + str(distance_real))
    print('p value = ' + str(p_val))
    print('n events = ' + str(sum(boundary_vector_gpt)+1))
    # for ee in events:
    #     print(ee[0:15])
    #     print(ee[-15:])
    #%%
    ax = sb.violinplot(rand_dists, orient='v', inner='points', jitter=True, color='lavender')
    sb.regplot(x=np.array([-0.2]), y=np.array([distance_real]), scatter=True, fit_reg=False, marker='o',
                 scatter_kws={"s": 10}, color = 'red', ax = ax).set(
                 title='Distance to bounds. P-val = ' + str(p_val))# fs = 10  # fontsize
     # sb.regplot(x=np.array([-0.2]), y=np.array([distance_12]), scatter=True, fit_reg=False, marker='o',
     #             scatter_kws={"s": 10}, color = 'blue', ax = ax)
    ax.set(xlabel='', ylabel='hamming distance')# pos = [1]

