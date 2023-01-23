#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 13:43:24 2022

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
from fastdtw import fastdtw

from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2')  # or any other checkpoint
word_embeddings = model.transformer.wte.weight  # Word Token Embeddings 
# token =  198
# tst = word_embeddings[token]


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
def split_interval (onset, offset, n):
    splits = np.linspace(onset, offset, n+1)
    return splits[0:-1], splits[1:]
def expand_onsets(words, onsets, offsets):
    #% convert word_onsets to token onsets
    word_tokens = []
    t_onsets = []
    t_offsets = []
    for windex, word in enumerate(words):
        wt = tokenizer(word)['input_ids']
        if len(wt) >1:
            onsets_new, offsets_new = split_interval(onsets[windex],
                                                     offsets[windex], len(wt))
        else:
            onsets_new  = onsets[windex]
            offsets_new = offsets[windex]
        word_tokens.append(wt)
        t_onsets.append(onsets_new)
        t_offsets.append(offsets_new)
    return word_tokens, t_onsets, t_offsets
#===============
# in case we want to compute the cost matrix of DTW 
# credit: (https://ealizadeh.com/blog/introduction-to-dynamic-time-warping/)
def compute_euclidean_distance_matrix(x, y) -> np.array:
    """Calculate distance matrix
    This method calcualtes the pairwise Euclidean distance between two sequences.
    The sequences can have different lengths.
    """
    dist = np.zeros((len(y), len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            dist[i,j] = sum((x[j]-y[i])**2)
    return dist


def compute_accumulated_cost_matrix(x, y) -> np.array:
    """Compute accumulated cost matrix for warp path using Euclidean distance
    """
    distances = compute_euclidean_distance_matrix(x, y)

    # Initialization
    cost = np.zeros((len(y), len(x)))
    cost[0,0] = distances[0,0]
    
    for i in range(1, len(y)):
        cost[i, 0] = distances[i, 0] + cost[i-1, 0]  
        
    for j in range(1, len(x)):
        cost[0, j] = distances[0, j] + cost[0, j-1]  

    # Accumulated warp path cost
    for i in range(1, len(y)):
        for j in range(1, len(x)):
            cost[i, j] = min(
                cost[i-1, j],    # insertion
                cost[i, j-1],    # deletion
                cost[i-1, j-1]   # match
            ) + distances[i, j] 
            
    return cost
#=============
# credit: https://www.geeksforgeeks.org/longest-repeating-and-non-overlapping-substring/
def longestRepeatedSubstring(str):
 
    n = len(str)
    LCSRe = [[0 for x in range(n + 1)]
                for y in range(n + 1)]
 
    res = "" # To store result
    res_length = 0 # To store length of result
 
    # building table in bottom-up manner
    index = 0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
             
            # (j-i) > LCSRe[i-1][j-1] to remove
            # overlapping
            if (str[i - 1] == str[j - 1] and
                LCSRe[i - 1][j - 1] < (j - i)):
                LCSRe[i][j] = LCSRe[i - 1][j - 1] + 1
 
                # updating maximum length of the
                # substring and updating the finishing
                # index of the suffix
                if (LCSRe[i][j] > res_length):
                    res_length = LCSRe[i][j]
                    index = max(i, index)
                 
            else:
                LCSRe[i][j] = 0
 
    # If we have non-empty result, then insert
    # all characters from first character to
    # last character of string
    if (res_length > 0):
        for i in range(index - res_length + 1,
                                    index + 1):
            res = res + str[i - 1]
 
    return res
#===============
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]
#%% versions and parameters


story_index = 0 # Monkey', 'Tunnel', 'Pieman'
version_index = 0 # 'long ' ,''
n_iter = 6
stories = ['Monkey', 'Tunnel', 'Pieman']

versions = ['long ' ,'', 'None ']

story = stories[story_index]   
version = versions[version_index] #"long " # "long " (include the space!)
    
text_plain = get_plain_text(story)
wl = Word_Loader(story, text_plain)
words, w_onsets, w_offsets = wl.load_words_and_times()
all_text = get_plain_text(story)
min_check_length = len(longestRepeatedSubstring(all_text))+1
#%%
time_courses = [];

for iteration in range(n_iter):
    
    bl = Boundary_Loader(story)
    ol = Output_Loader(story,version, n_iter)
    
    boundary_vector_gpt = ol.load_bound_vec(iteration)
    events = ol.load_event_list(iteration)
    responses = ol.load_response_list(iteration)

    
    NL_probs  = []
    TOK_items  = []
    ListTOKENS = []
    ListONSETS = []
    ListOFFSETS = []
    
    start_index = 0
    
    
    #r = 0; resp = responses[0]
    for r, resp in enumerate(responses):
    
        # get the response text
        response_text = resp['choices'][0]['text']
        
        # only at the beginning
        if r == 0:
            #trim leading new line!
            while response_text[0] == '\n':
                response_text = response_text[1:]
        #i f the last event doesn't count
        if r < len(responses):
            # also trim trailing new line
            while response_text[-1] == '\n':
                response_text = response_text[:-1]
    
        # if we are not in the last segment....
        if r < len(responses)-1:
            
            # === FIRST, WE WANT TO KNOW WHAT PART OF THE TEXT WE ARE IN 
            # we need to find the start of the last event 
            # (or last sentence for copy instruction)
            if iteration == 'NONE':
                last_index = response_text.rfind(".")
            else:
                last_index = response_text.rfind("\n")
        
            #... and work with only part of the response text 
            current_response_segment =   (response_text[:last_index+1])
            
            # === SECOND, WE WANT TO KNOW WHAT PART OF THE WORD LIST APPLIES HERE 
    
            # so pull out the end segment...
            end_segment =   (response_text[last_index+1:]).replace('"', '')
             
            # ... then  temporarily join all the words
            tmp_string = " ".join(words).casefold()
            
            # and find out, where in the word list the end of the response starts
            # and find the start of this segment in the word_list
            list_start =  tmp_string.rfind(end_segment[0:min_check_length].casefold())
            # we get the number of tail words from the tokenization of the end segment
            n_tailwords = len(word_tokenize(tmp_string[list_start:]))
            
    
            # now pull out all the words and time stamps from
            # the alignment list that are in the current segment
            words_tmp = words[start_index:-n_tailwords]
            onsets_tmp = w_onsets[start_index:-n_tailwords]
            offsets_tmp = w_offsets[start_index:-n_tailwords]
            
            # and set the new LIST start index for the next loop
            start_index = start_index + len(words_tmp)
            
        # in the final sement we can just take the full response text and 
        # all remaining words from the list
        else:
            current_response_segment = response_text
            words_tmp = words[start_index:]
            onsets_tmp = w_onsets[start_index:]
            offsets_tmp = w_offsets[start_index:]
        
        # from before, we expand the tokens and onsets from the onset word list
        # (there may be 2-3 tokens in a word)
        curr_list_wt, curr_onset, curr_offset = expand_onsets(words_tmp, 
                                                              onsets_tmp, 
                                                              offsets_tmp)
        # === THIRD, WE WANT TO PULL OUT log-probs and tokens
    
        tmp = resp['choices'][0]['logprobs']['top_logprobs']
    
        # restrict only to those tokens that are part of the current segment
        current_tokens = tokenizer(current_response_segment)['input_ids']
        tmp = tmp[:len(current_tokens)]
        
        # now we collect the tokens from the response in items
        items = []
        # and the associated log-probs in nl_probs
        nl_probs = []
        for ttind, tt in enumerate(tmp):
            # keys are the items
            keys = list(tt.keys())
            # values are the logprobs
            values = list(tt.values())
            # append the token of the maximum probability item
            items.append(
                tokenizer(keys[np.argmax(values)])['input_ids'])
            # if there is a newline among the keys, we store its probability
            if "\n" in keys:
                nl_probs.append(values[keys.index("\n")])
            # otherwise, we just append a NaN
            else: 
                nl_probs.append(np.nan)
        # == and then keep everything for later!     
        NL_probs  =  NL_probs + nl_probs
        TOK_items = TOK_items + items
    
        ListTOKENS = ListTOKENS   + curr_list_wt
        ListONSETS = ListONSETS   + curr_onset
        ListOFFSETS = ListOFFSETS + curr_offset
    
    
    
    #% To compute the distances we need to flatten everything
    flat_list_wt = [wt for sublist in ListTOKENS for wt in sublist]
    flat_onset  = [co for sublist in ListONSETS for co in sublist]
    flat_onset = [co.flatten() for co in flat_onset]
    flat_offset = [co for sublist in ListOFFSETS for co in sublist]
    flat_offset = [co.flatten() for co in flat_offset]
    flat_onset  = np.array(flat_onset)
    flat_offset  = np.array(flat_offset)
    
    #% now we get the gpt-2 word embeddings to compute a warp path
    resp_emb = [];
    for t in TOK_items:
        resp_emb.append(word_embeddings[t].detach().numpy().flatten())
    list_emb = [];
    for i, t in enumerate(flat_list_wt):
        list_emb.append(word_embeddings[t].detach().numpy().flatten())
    
    #cost_matrix = compute_accumulated_cost_matrix(list_emb, resp_emb)
    
    dtw_distance, warp_path = fastdtw(list_emb, resp_emb, dist = sp.spatial.distance.euclidean)
    
    path_x = [p[0] for p in warp_path]
    path_y = [p[1] for p in warp_path]
    
    
    token_onsets_fin = flat_onset[path_x]
    token_offsets_fin = flat_offset[path_x]
    NL_probs_arr = np.array(NL_probs)
    nl_vals_fin = NL_probs_arr[path_y]
    #%
#    word_embeddings = None
    
    #%
    # get a time-course of button presses
    if story == "Monkey":
        bp_all = bl.load_button_presses()
        bps = np.concatenate((np.average(bp_all[0], axis = 0), 
                                 np.average(bp_all[1], axis = 0), 
                                 np.average(bp_all[2], axis = 0), 
                                 np.average(bp_all[3], axis = 0)))
    else:
        bps = np.average(bl.load_button_presses()[0], axis = 0)
    
    multiplier = np.nan
    if story == "Tunnel":
        multiplier = 10
    else: 
        multiplier = 1000
    # to take care of overlap in token onsets, we just put everything into a big 
    # nan-matrix and then compute the nan-save average
    #big_mat = np.ones((len(nl_vals_fin), len(bps)))*np.nan
    if story == "Tunnel":
        print('interpolate nans in time courses')
        nans, x = nan_helper(token_onsets_fin)
        token_onsets_fin[nans] = np.interp(
            x(nans), x(~nans), token_onsets_fin[~nans])
        nans, x = nan_helper(token_offsets_fin)
        token_offsets_fin[nans] = np.interp(
            x(nans), x(~nans), token_offsets_fin[~nans])
    
    # add up the values here
    val_vec =  np.zeros(( len(bps)))
    # add up the divisors here
    div_vec = np.zeros(len(bps))
    # make a vector at the time-scale of milliseconds 
    for oind, ons in enumerate(token_onsets_fin):
        oind
        mson = int(np.round(ons*multiplier)) # onset in ms vale
        msoff = int(np.round(token_offsets_fin[oind]*multiplier)) # offset in ms
        if not(np.isnan(nl_vals_fin[oind])): # if onset val is not nan
            val_vec[mson:msoff] = val_vec[mson:msoff] +  nl_vals_fin[oind] # add val
            div_vec[mson:msoff] = div_vec[mson:msoff] + 1 # count n 
    time_course_fin = val_vec # the values
    # need to be divided by the sum whenever more than one value was added
    time_course_fin[np.where(div_vec>1)] = (val_vec[np.where(div_vec>1)] /
                                            div_vec[np.where(div_vec>1)])
    time_course_fin[np.where(div_vec==0)] = np.nan # if no value was added -> nan
    #time_course_fin = np.nanmean(big_mat, axis = 0)
    #big_mat = None
    #%
    time_to_onset = int(np.ceil(w_onsets[0]*multiplier))-1
    time_course_fin[:time_to_onset] = np.nanmin(time_course_fin)
    # and the last 1 seconds
    #time_course_fin[-1000:] = np.nanmin(time_course_fin)
    
    # now interpolate the nans
    nans, x = nan_helper(time_course_fin)
    time_course_fin[nans] = np.interp(x(nans), x(~nans), time_course_fin[~nans])
    time_courses.append(time_course_fin)

#%%
corrs = []
# %%
if iteration >0:
    time_course_fin = np.nanmean(np.array(time_courses), axis = 0)
time_course_fin = sp.stats.zscore(time_course_fin)




if story == "Pieman":
    run_id = 1 
    bound_dat = np.average(bl.load_button_presses()[run_id], axis = 0)

else:
    bound_dat = bps

log_bps = bound_dat.copy()
log_bps[np.where(bound_dat ==0)] = np.NAN
nans, x = nan_helper(log_bps)

log_bps = np.log(bound_dat)
log_bps[nans] = np.interp(x(nans), x(~nans), log_bps[~nans])
bound_dat = sp.stats.zscore(log_bps)

#%% cross correlate
corr = sp.signal.correlate(bound_dat, time_course_fin, mode = 'same')/len(time_course_fin)
p = sp.stats.pearsonr(bound_dat, time_course_fin, alternative='two-sided')

lags = sp.signal.correlation_lags(len(time_course_fin), len(bound_dat), mode = 'same')
lags[np.argmax(corr)]
corr[np.where(lags==0)][0]
max(corr)

print("the maximum correlation is " + str( round(max(corr), 3)))

corrs.append(corr)
#%%

fig, axes = plt.pyplot.subplots(2, 1, sharex=True)
axes[0].plot(time_course_fin)
axes[0].set_title('Log_prob')
axes[1].plot(bound_dat)
axes[1].set_title('Behavior')

#%%

sns.set_style('white')
sns.set_context("paper", font_scale = 2)

window_width = 20000 # 20 seconds

xdata = lags[
    np.where(np.logical_and(lags>-window_width , lags <window_width))]/1000


ydata1 = corrs[0][
    np.where(np.logical_and(lags>-window_width , lags <window_width))]

ydata2 = corrs[1][
    np.where(np.logical_and(lags>-window_width , lags <window_width))]
fig, ax = plt.pyplot.subplots()
line1 = ax.plot(xdata, ydata1,  label="Behav. Run 1", color = "blue")
ax.set_xlabel("Lag (seconds)")
ax.set_ylabel("Pearson r")
line2 = ax.plot(xdata, ydata2,  label="Behav. Run 2", color = "red")

ax.legend(prop = {'size' : 14},
           loc = 'upper right', shadow = True)

sns.despine(right = True)
fig.show()
fig.savefig('PiemanShortXcorr.svg')
#%%


sns.set_style('white')
sns.set_context("paper", font_scale = 2)

window_width = 50000 

xdata = lags[
    np.where(np.logical_and(lags>-window_width , lags <window_width))]/1000


ydata1 = corrs[0][
    np.where(np.logical_and(lags>-window_width , lags <window_width))]

fig, ax = plt.pyplot.subplots()
line1 = ax.plot(xdata, ydata1,  color = "blue")
ax.set_xlabel("Lag (seconds)")
ax.set_ylabel("Pearson r")


sns.despine(right = True)
fig.show()
fig.savefig('MonkeyLongXcorr.svg')



