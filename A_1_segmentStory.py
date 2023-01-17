
"""
Created on Fri Nov  4 15:46:14 2022

@author: sebastian
"""
# NOTES: It is best to keep the max output tokens to a minimum. Otherwise the model is more likely to diverge from the exact copy
# davinci-002 is less likely to diverge from the exact copy than 003
# the tunnel story occasionally has a lot of events and therefore might end the output before the story ends; here it helps to add 512 tokens of extra padding
# Using too much of the context window is bad
import math
import numpy as np
from transformers import GPT2TokenizerFast
import pickle   
import os
import openai
from nltk.tokenize import sent_tokenize


def get_apiKey():
    home = os.path.expanduser("~")   
    fp = open(home + '/' + 'openai.txt',
               encoding='utf-8-sig')
    return fp.read().replace('\n', '')

# from https://www.geeksforgeeks.org/longest-repeating-and-non-overlapping-substring/
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
 
#%% versions and parameters
n_iter = 6
story_index = 1 # Monkey', 'Tunnel', 'Pieman'
version_index = 1 # 'long ' ,''
iteration  = 3
padding = 0# 512 # added for tunnel...

#%%

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

stories = ['Monkey', 'Tunnel', 'Pieman']
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
openai.api_key = get_apiKey()


versions = ['long ' ,'']
#%%
story = stories[story_index]   
version = versions[version_index] #"long " # "long " (include the space!)
# don't use all the context window.
# max context window is 4096
context_window_width = 2048 #4096


text_file_name = os.pardir + '/' + story + '/' + story + 'Clean'
#%% read in the text
#import matplotlib.pyplot as plt
fp = open(os.pardir + '/' +  story  + '/' +  story + "Clean.txt",
         encoding='utf-8-sig')
data = fp.read()
# make sure there are no quotation marks (there shouldn't be any)
all_text = data.replace('\\"', '')
# three dots are also confusing, because nltk doesn't see them as new sentence
all_text = data.replace('...', '.')

# tokenize the text to sentences for sentence level boundaries
all_sentences = sent_tokenize(all_text)

min_check_length = len(longestRepeatedSubstring(all_text))+1

# define prompt and compute sizes 
prompt_text = ("An event is an ongoing coherent situation. The following story needs to be copied and segmented into " 
               + version 
               +  "events. Copy the following story word-for-word and start a new line whenever one " 
               + version 
               +  "event ends and another begins. This is the story: ")
prompt_suffix = "\n This is a word-for-word copy of the same story that is segmented into " + version +  "events: "


# number of tokens that are used for the prompt
n_prompt_tokens = len(tokenizer(prompt_text + prompt_suffix)['input_ids'])
# maximum amount of tokens that are left for the story
max_n_story_tokens = math.floor((context_window_width - n_prompt_tokens)/2)
#%% run iterations of the same instructions

#for iteration in range(n_iter):
print(iteration)
#% file name definition
output_file_name1 = (os.pardir + '/' + story +  '/outputs/' + story  + 
                     '_iter_' + str(iteration) +  
                     '_version_' + version[:-1] + '_Responses' )
output_file_name2 = (os.pardir + '/' + story +  '/outputs/' + story  + 
                     '_iter_' + str(iteration) +  
                     '_version_' + version[:-1] + '_Events' )

output_file_name3 = (os.pardir + '/' + story +  '/outputs/' + story  + 
                     '_iter_' + str(iteration) +  
                     '_version_' + version[:-1] + '_Boundary_Vector' )    

# don't overwrite!
assert(not(os.path.isfile(output_file_name1 + '.pkl')))



tokens = tokenizer(all_text)['input_ids']


# length of the story in tokens
n_storytokens = len(tokens)

# max tokens for a call to openai
n_max_token = context_window_width -(max_n_story_tokens+n_prompt_tokens)


# maximum allowed loops (a stuck while loop should never use tokens)
# set to a reasonable number (no more than twice the whole story when incl. 
# overlap)
max_loop = np.ceil(8*n_storytokens/n_max_token)
# loop counter
i_loop = 0
# start index for story parsing
start_index = 0
# collect output from the model
responses = []
# collect a list of events in text form
event_list = []
finished = False
#%%
# loop if/while we are not finished yet
while not(finished):
    # count loops to break after max loops 
    print(i_loop)
    i_loop = i_loop + 1
    if i_loop >= max_loop:
        print('too many loops')
        openai.api_key = -1
        finished = True
        break
    # tokenize everything from the current start index
    tokens = tokenizer(all_text[start_index:])
    # total number of tokens in the remaining story
    n_storytokens = len(tokens[0])
   
    # get the actual token IDs
    tokens = tokens['input_ids']
    
    # if we have less tokens left than maximally allowed, we adjust the max
    if n_storytokens < max_n_story_tokens:
        max_n_story_tokens = n_storytokens;
    
    # the new tokens are the maximally allowed tokens from remaining ones
    new_tokens= tokens[0:max_n_story_tokens]
    
    # the decode all tokens and join them to a text of the current segment
    text_segment = ''.join(
        [tokenizer.decode(x) for x in new_tokens])
    
    # now we get the completion from gpt-3 
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt= prompt_text + text_segment + prompt_suffix + " \n",
      temperature=0,
      max_tokens=n_max_token+padding,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      logprobs = 5
    )
    # we collect the response
    responses.append(response)
    # and pull out the text answer
    response_text = response['choices'][0]['text']
    # only if there are trailing newline characters... trim them
    while response_text[-1] == '\n':
        response_text = response_text[:-1]
    
    # splitting at new line results in an event list
    events = response_text.splitlines()  # , everyone, a
    # remove empty events (double new-line)
    events_trimmed = [i for i in events if i]

    # we are finished when the length of the remaining tokens (start to end
    # ) is the same as the length of the tokens we just segmented
    finished = len(tokens)==len(new_tokens)
 
    #... but if we are not finished, we want to discard the final event 
    if not(finished):
        # remove the final event
        events_trimmed = events_trimmed[:-1]
        
    # the event list collects all the events that we have so far
    event_list = event_list + events_trimmed
    
    # the final new line is the index where the last event starts
    last_index = response_text.rfind("\n")
    
    # the end segment is not a full sentence even..
    # ... also remove potential quotation marks (they cause confusion)
    end_segment =   (response_text[last_index+1:]).replace('"', '')
    
    # find 25 chars of the end segment in the text to define the new start
    # it might happen that a dash appears before the event... 
    if end_segment[0] == '-':
            start_index = all_text.rfind(end_segment[1:min_check_length+1])
    # or not...
    else:
            start_index = all_text.rfind(end_segment[0:min_check_length])
    
    # try to fix it if it's a minor thing
    if start_index == -1:
        if end_segment[0] == '-':
                tmp_index = all_text.rfind(end_segment[26:min_check_length+26])
        # or not...
        else:
                tmp_index = all_text.rfind(end_segment[25:min_check_length+25])
        if tmp_index != -1:
            start_index = tmp_index - 25
            

    # if the event is not in the text, we need to stop 
    # ..this is not a good way to end, but may be correct???
    if start_index == -1:
        print('warning.. the final segment was not found in the text')
        openai.api_key = -1
        finished = True

    #new_tokens = tokenizer(lines[0][last_index:last_index+max_n_story_tokens])

#%%  now find the indices of the sentence bounds that are event boundaries
   
first_sen = [] # this is for debugging, store the first sentences of events
n_sent = []
# boundary vec is between each sentence (0 or 1)
boundary_vec = np.zeros(len(all_sentences)-1,)
# start before the first sentence
idx = -1;
# loop the event list
for ei, ee in enumerate(event_list):
    # first sentence in the event... 
    first_sen.append(all_sentences[idx+1])
    
    # collect number of sentences in the current event (safer than words)
    ss = sent_tokenize(ee)
    # add the number of sentences to the helper index
    idx = idx + len(ss)
    n_sent.append(len(ss))
    # if the index is before the final sentence, we set the bound to 1
    if idx < len(boundary_vec): 
        boundary_vec[idx] = 1
#%% compare to backwards for debugging!!
# we do the same process in a backwards direction
boundary_vec_compare = np.zeros(len(all_sentences)-1,)
#idx starts at the end
idx = len(all_sentences)-1
# loop through events in reversed order
for ee in reversed(event_list):
    # and count the number of sentences
    ss = sent_tokenize(ee)
    idx = idx - len(ss)
    # if we are after the first sentence, we set the bound to 1
    if idx >-1: 
        boundary_vec_compare[idx] = 1
#%% for sanity check: 
# the number of boundaries needs to be equal to the number of events -1 
assert((sum(boundary_vec) == (len(event_list)-1)))
# the backward and forward solution need to be identical!
assert((boundary_vec == boundary_vec_compare).all())

#%% save the result with pickle:
with open(output_file_name1 + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(responses, f)
with open(output_file_name2 + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(event_list, f)
    
with open(output_file_name3 + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(boundary_vec, f)

