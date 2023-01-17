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
import seaborn as sns
import matplotlib as plt
from loaders import Boundary_Loader, Output_Loader, Word_Loader
from pingouin import ttest
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
#for si in range(3):
story_index = 0 # Monkey', 'Pieman'
# version_index = 1 # 'long ' ,''
n_iter = 6
stories = ['Monkey', 'Pieman']

versions = ['long ' ,'']

story = stories[story_index]   
# version = versions[version_index] #"long " # "long " (include the space!)


text_plain = get_plain_text(story)

bl = Boundary_Loader(story)
ol = Output_Loader(story,versions[1], 0)
ol_long = Output_Loader(story,versions[0], 0)

wl = Word_Loader(story, text_plain)

boundary_vector_gpt = ol.load_bound_vec(0)
boundary_vector_gpt_long = ol_long.load_bound_vec(0)

events = ol.load_event_list(0)
responses = ol.load_response_list(0)

words, w_onsets, w_offsets = wl.load_words_and_times()
#% get the human boundaries too
# get the sentence bounds
sentence_bounds, sentences, onset_list, offset_list, starting_words, ending_words = (
    get_sentence_bounds(text_plain))
#%%
# Note: bounds 2 is for Pieman
Bounds_all1, Bounds_all2  = bl.load_human_bounds_ms()

#%% get individual bound 

# to collect sentence-bound vectors for each subject
bound_arrays = []

# for MIM we need to know the offsets in the audio for segments
cms = np.cumsum(np.array([0, 444455,415936,482353, 462256]))

# get the individual button presses frrom file
bound_ars = bl.load_button_presses()

# delete empty outputs
while any(elem is None for elem in bound_ars):
    bound_ars = bound_ars[:-1]

# loop throuhgh the button press groups
for indx, bar in enumerate(bound_ars):
    
    # create an empty sentence vector matrix (subject x sentences)
    boundary_array = np.zeros((len(sentences)-1,np.shape(bar)[0]))
    # in MIM there are offsets
    sect_offset = 0
    if story == "Monkey": # for monkey
        sect_offset = cms[indx]/1000
    #loop through subjects
    for sj in np.arange(np.shape(bar)[0]):
        # get the event boundaries in seconds
        if story  == 'Tunnel':
            esec = np.asarray(np.where(np.array(bar[sj,])))/10 + sect_offset

        else:
            esec = np.asarray(np.where(np.array(bar[sj,])))/1000 + sect_offset
        
        # somehow the output is in an array
        bounds = esec[0]
        # now loop through the boundaries in seconds
        for bound in bounds:
            # and set the corresponding sentence boundary to True
            boundary_array[np.absolute(
                sentence_bounds - bound).argmin()][sj]  = True
    # collect all the sub-arrays (sample in MIM, run in Pieman)
    bound_arrays.append(boundary_array)
        
#%% compute the average and range for behavioral responses for description
for bri in bound_arrays:
    np.sum(bri, axis = 0)    
    print(story + ": the min number of button presses is " + 
          str(min( np.sum(bri, axis = 0))) + " the max number of button presses is "
          + str(max( np.sum(bri, axis = 0))))
    print(story + ": the average number of button presses is " + 
          str(np.average( np.sum(bri, axis = 0))) + " std: "
          + str(np.std( np.sum(bri, axis = 0))))

#%% create a joint solution for Monkey by stitching together random groups 
np.random.seed(0)

if story == "Monkey":
    # find the minimum amount of people per sample
    min_p  = np.Inf
    for b in bound_arrays:
        if b.shape[1] < min_p:
            min_p = b.shape[1]
    # create a new boundary vector array 
    b_arr_monkey = np.zeros((bound_arrays[0].shape[0],min_p))
    # now go through all the 4 boundary arrays
    for midx, bound_array in enumerate(bound_arrays):
        # take the indices where the bounds are in that segment
        idx4monkey = np.where(np.logical_and(
          sentence_bounds[:,] >=cms[midx]/1000,
          sentence_bounds[:,] <cms[midx+1]/1000))[0]
        # take a random selection of 30 people
        sel = np.random.choice(bound_array.shape[1], size=min_p, replace=False)
        # those are the current segments
        bound_array_segment = bound_array[:,sel]
        # now set the new array to those values
        b_arr_monkey[idx4monkey,:] = bound_array_segment[idx4monkey,:]
    
    # overwrite previous version
    bound_arrays = []
    bound_arrays.append(b_arr_monkey)


#%% distance to average solution

sentence_bounds_ms_gpt = []
sentence_bounds_ms_gpt_long = []

array_names = ['array 1', 'array 2']

bounds_all_s = bl.load_human_bounds_ms()
subject_dist_arrays_avg = []
gpt_dist_arrays_avg = []
gpt_dist_arrays_avg_long = []
# loop throuh the runs
for seg_index, bound_array in enumerate(bound_arrays):
    
    boundary_vecX = get_human_boundary_vector(sentence_bounds, 
                                               bounds_all_s[seg_index]/1000)
    # collect the average distances (hamming distances between each subject and
    # the consensus solution)
    
    # First, per subject
    avg_dists = np.zeros(np.shape(bound_array)[1])
    for sj in np.arange(np.shape(bound_array)[1]):
        b_vec = bound_array[:,sj]
        avg_dists[sj] = sp.spatial.distance.hamming(
            b_vec, boundary_vecX, w=None)
    subject_dist_arrays_avg.append(avg_dists)
    
    # Second, per GPT iteration
    avg_dists_gpt = np.zeros(n_iter)
    avg_dists_gpt_long = np.zeros(n_iter)

    for iteration in range(n_iter): 
        boundary_vector_gpt = ol.load_bound_vec(iteration)
        sentence_bounds_ms_gpt.append(
            sentence_bounds[np.where(boundary_vector_gpt)])
        avg_dists_gpt[iteration] = sp.spatial.distance.hamming(
            boundary_vector_gpt, boundary_vecX
            , w=None)
    for iteration in range(n_iter): 
        boundary_vector_gpt_long = ol_long.load_bound_vec(iteration)
        sentence_bounds_ms_gpt_long.append(
            sentence_bounds[np.where(boundary_vector_gpt_long)])
        avg_dists_gpt_long[iteration] = sp.spatial.distance.hamming(
            boundary_vector_gpt_long, boundary_vecX
            , w=None)
    gpt_dist_arrays_avg.append(avg_dists_gpt)
    gpt_dist_arrays_avg_long.append(avg_dists_gpt_long)
#%%

mdic = {"array_long":sentence_bounds_ms_gpt_long, "array_short": sentence_bounds_ms_gpt}

sp.io.savemat("sentence_bounds_pieman_gpt.mat", mdic)
#%% Now test the difference for significance with a T-test

b_ind = 0
T = ttest(
    subject_dist_arrays_avg[b_ind], gpt_dist_arrays_avg[b_ind])


print(array_names[b_ind] + " - The average distance to human consensus is: " +str(
    round(np.average(subject_dist_arrays_avg[b_ind]),3)) +  " +-SD=" + str(
        round(np.std(subject_dist_arrays_avg[b_ind]),3)) + " for humans and " +str(
            round(np.average(gpt_dist_arrays_avg[b_ind]),3)) +  " +-SD=" + str(
                round(np.std(gpt_dist_arrays_avg[b_ind]),3)) + 
        " for GPT-3 (p = " + str(
            T["p-val"].to_numpy()) + ") The BF of this difference is " + str(
                T["BF10"].to_numpy()))
                

ax = sns.violinplot(subject_dist_arrays_avg[b_ind], orient='v',
                   inner='points', jitter=True, color='lavender')
for iteration in range(n_iter):
    sns.regplot(x=np.array([-0.2]), y=np.array(
        [gpt_dist_arrays_avg[b_ind][iteration]]), 
        scatter=True, fit_reg=False, marker='o',
                scatter_kws={"s": 10}, color = 'red', ax = ax).set(
                    title= story + ': Distance to average, version '+ str(b_ind+1))# fs = 10  # fontsize
                
ax.set(xlabel='', ylabel='average hamming distance')# pos = [1]

#%% Also test for long events

b_ind = 1
T = ttest(
    subject_dist_arrays_avg[b_ind], gpt_dist_arrays_avg_long[b_ind])


print(array_names[b_ind] + " - The average distance to human consensus is: " +str(
    round(np.average(subject_dist_arrays_avg[b_ind]),3)) +  " +-SD=" + str(
        round(np.std(subject_dist_arrays_avg[b_ind]),3)) + " for humans and " +str(
            round(np.average(gpt_dist_arrays_avg_long[b_ind]),3)) +  " +-SD=" + str(
                round(np.std(gpt_dist_arrays_avg_long[b_ind]),3)) + 
        " for GPT-3 (p = " + str(
            T["p-val"].to_numpy()) + ") The BF of this difference is " + str(
                T["BF10"].to_numpy()))
                

ax = sns.violinplot(subject_dist_arrays_avg[b_ind], orient='v',
                   inner='points', jitter=True, color='lavender')
for iteration in range(n_iter):
    sns.regplot(x=np.array([-0.2]), y=np.array(
        [gpt_dist_arrays_avg_long[b_ind][iteration]]), 
        scatter=True, fit_reg=False, marker='o',
                scatter_kws={"s": 10}, color = 'red', ax = ax).set(
                    title= story + ': Distance to average, set '+ str(b_ind+1))# fs = 10  # fontsize
                
ax.set(xlabel='', ylabel='average hamming distance')# pos = [1]

#%% for Pieman, draw split violins:

sns.set_style('white')
sns.set_context("paper", font_scale = 2)
    
import pandas as pd

df1 = pd.DataFrame({'Story': 'Pieman', 'Run': 'run 1' , 'hamming distance': subject_dist_arrays_avg[0]})
df2 = pd.DataFrame({'Story': 'Pieman','Run': 'run 2' , 'hamming distance': subject_dist_arrays_avg[1]})

df = pd.concat((df1, df2), axis = 0)

fig, ax = plt.pyplot.subplots()

sns.violinplot(data=df,x = 'Story', y = "hamming distance",hue = "Run", split=True, linewidth=1,
                   inner='box', color='lavender', ax = ax)

b_ind = 0;
for iteration in range(n_iter):
    offst = (np.random.rand()-0.5)/4
    print(offst)
    sns.regplot(x=np.array([offst-0.2]), y=np.array(
        [gpt_dist_arrays_avg[b_ind][iteration]]), 
        scatter=True, fit_reg=False, marker='o',
                scatter_kws={"s": 4}, color = (53/255, 206/255, 141/255), ax = ax).set(
                    title= story + ': Distance to consensus')# fs = 10  # fontsize

b_ind = 1;
                    
for iteration in range(n_iter):
    offst = (np.random.rand()-0.5)/4
    print(offst)
    sns.regplot(x=np.array([offst+0.2]), y=np.array(
        [gpt_dist_arrays_avg[b_ind][iteration]]), 
        scatter=True, fit_reg=False, marker='o',
                scatter_kws={"s": 4}, color = (53/255, 206/255, 141/255), ax = ax).set(
                    title= story + ': Distance to consensus')# fs = 10  # fontsize

b_ind = 0;
for iteration in range(n_iter):
    offst = (np.random.rand()-0.5)/4
    print(offst)
    sns.regplot(x=np.array([offst-0.2]), y=np.array(
        [gpt_dist_arrays_avg_long[b_ind][iteration]]), 
        scatter=True, fit_reg=False, marker='o',
                scatter_kws={"s": 4}, color = (244/255, 162/255, 89/255), ax = ax).set(
                    title= story + ': Distance to consensus')# fs = 10  # fontsize

b_ind = 1;
                    
for iteration in range(n_iter):
    offst = (np.random.rand()-0.5)/4
    print(offst)
    sns.regplot(x=np.array([offst+0.2]), y=np.array(
        [gpt_dist_arrays_avg_long[b_ind][iteration]]), 
        scatter=True, fit_reg=False, marker='o',
                scatter_kws={"s": 4}, color = (244/255, 162/255, 89/255), ax = ax).set(
                    title= story + ': Distance to consensus')# fs = 10  # fontsize

fig.savefig('PiemanVioAll2.svg')

#%% monkey
b_ind = 0
sns.set_style('white')
sns.set_context("paper", font_scale = 2)

fig, ax = plt.pyplot.subplots()

ax = sns.violinplot(subject_dist_arrays_avg[b_ind], orient='v',
                   inner='box', jitter=True, color='lavender', ax = ax)

sns.regplot(x=np.array([0]), y=np.array([np.average(subject_dist_arrays_avg[b_ind])]), scatter=True, fit_reg=False, marker='o',
            scatter_kws={"s": 18},color =  (255/255, 0/255, 0/255),)# fs = 10  # fontsize


for iteration in range(n_iter):
    offst = (np.random.rand()-0.5)/4
    print(offst)
    sns.regplot(x=np.array([offst-0.2]), y=np.array(
        [gpt_dist_arrays_avg[b_ind][iteration]]), 
        scatter=True, fit_reg=False, marker='o',
                scatter_kws={"s":  14}, color =  (53/255, 206/255, 141/255), ax = ax).set(
                    title= story + ': Distance to consensus, set' + str(b_ind+1))# fs = 10  # fontsize

for iteration in range(n_iter):
    offst = (np.random.rand()-0.5)/4
    print(offst)
    sns.regplot(x=np.array([offst-0.2]), y=np.array(
        [gpt_dist_arrays_avg_long[b_ind][iteration]]), 
        scatter=True, fit_reg=False, marker='o',
                scatter_kws={"s": 14}, color = (244/255, 162/255, 89/255), ax = ax).set(
                    title= story + ': Distance to consensus,  set' + str(b_ind+1))# fs = 10  # fontsize

ax.set(xlabel='', ylabel='average hamming distance')# pos = [1]


ax.set(xlabel='', ylabel='average hamming distance')# pos = [1]


sns.despine(right = True)
fig.show()
#%%
fig.savefig('PiemanVioAll31.svg')

fig.savefig('PiemanVioAll32.svg')

fig.savefig('MonkeyVioAll3.svg')
#%% this is legacy code: compute the distances pairwise
#%% END OF SCRIPT #%% END OF SCRIPT
#%% END OF SCRIPT
#%% END OF SCRIPT
#%% END OF SCRIPT
#%% END OF SCRIPT
#%% END OF SCRIPT
#%% END OF SCRIPT
#%% END OF SCRIPT
#%% END OF SCRIPT
#%% END OF SCRIPT
#%% 


#%% get average distances for each participant
#! note that the second dimension of input 2 needs to be the subject dimension
def average_dist(bound1, bounds2):
    tmpvec = np.zeros(np.shape(bounds2)[1])
    for sj in np.arange(np.shape(bounds2)[1]):
        tmpvec[sj] =  sp.spatial.distance.hamming(
            bound1, bounds2[:,sj], w=None)
        
    return np.average(tmpvec)
subject_dist_arrays = []
gpt_dist_arrays = []


for midx, bound_array in enumerate(bound_arrays):
    avg_dists = np.zeros(np.shape(bound_array)[1])
    
    # index only subpart of the sentenecs for the monkey podcast, because it 
    # was collected in 4 batches of different subjects
   
    for sj in np.arange(np.shape(bound_array)[1]):
        b_vec = bound_array[:,sj]
        barr  = np.concatenate(
            (bound_array[:,0:sj], 
             bound_array[:,sj+1:]), axis = 1)
        avg_dists[sj] = average_dist(b_vec,barr)
    subject_dist_arrays.append(avg_dists)
    
    avg_dists_gpt = np.zeros(n_iter)
    for iteration in range(n_iter): 
        boundary_vector_gpt = ol.load_bound_vec(iteration)
        avg_dists_gpt[iteration] = average_dist(
            boundary_vector_gpt,bound_array)
    gpt_dist_arrays.append(avg_dists_gpt)

#%%
b_ind = 1
T = ttest(
    gpt_dist_arrays[b_ind], subject_dist_arrays[b_ind])


print(array_names[b_ind] + " - The average distance to individual other human raters is: " +str(
    round(np.average(subject_dist_arrays[b_ind]),3)) +  " +-SD=" + str(
        round(np.std(subject_dist_arrays[b_ind]),3)) + " for humans and " +str(
            round(np.average(gpt_dist_arrays[b_ind]),3)) +  " +-SD=" + str(
                round(np.std(gpt_dist_arrays[b_ind]),3)) + 
        " for GPT-3 (p = " + str(
            T["p-val"].to_numpy()) + ") The BF of this difference is " + str(
                T["BF10"].to_numpy()))
                

ax = sns.violinplot(subject_dist_arrays[b_ind], orient='v',
                   inner='points', jitter=True, color='lavender')
for iteration in range(n_iter):
    sns.regplot(x=np.array([-0.2]), y=np.array(
        [gpt_dist_arrays[b_ind][iteration]]), 
        scatter=True, fit_reg=False, marker='o',
                scatter_kws={"s": 10}, color = 'red', ax = ax).set(
                    title=story + ': Distance to others, set ' + str(b_ind+1) )# fs = 10  # fontsize
                
ax.set(xlabel='', ylabel='average hamming distance')# pos = [1]

