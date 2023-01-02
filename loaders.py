#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 18:38:45 2022

@author: sebastian
"""
import pickle
import scipy as sp
import numpy as np
import re
from nltk.tokenize import word_tokenize

import os
class Boundary_Loader:
    def __init__(self, story):
        self.story = story
        self.path = os.pardir + '/' + story +  '/sourcedata/' 

            
    def get_Manoj_filtered_bounds_ms(self):
        """Currently hard coded, but chnage to read from a file"""
        human_filtered_bounds = []
        if self.story =='Pieman':
            human_boundaries2 = np.array([27118, 42021, 53883, 72791, 97259, 117622, 133494, 185887, 203434, 208573, \
                                         226955, 268269, 279488, 289554, 308024, 329378, 381957, 423366, 431517])
            human_filtered_bounds=human_boundaries2
            human_filtered_bounds = np.delete(human_boundaries2,(18))
            human_filtered_bounds=human_filtered_bounds
        elif self.story =='Monkey':
            human_boundaries = np.array([  21700,   69591,   98503,  146510,  175480,  234272,  284866, \
                315029,  386838,  387720,  387781,  388185,  388974,  442706, \
                541682,  541714,  650849,  832964,  908715,  926219, 1091764, \
               1167122, 1167208, 1315393, 1369944, 1444632, 1468867, 1468923, \
               1778312, 1797724])
            human_filtered_bounds = np.delete(human_boundaries, (8, 9,10, 11, 14,21,26, 29))
            human_filtered_bounds = human_filtered_bounds
        
            human_seg_90 = np.array([  21700,   69591,   98503 , 146510,  175480 , 183933 , 187968 , 234272 , 242959, \
                  243139,  243231 , 284866 , 296304 , 315029,  359647 , 386838 , 387207,  387720, \
                  387781 , 388185,  388974 , 442772 , 456868,  466100 , 466384 , 469985,  541682, \
                  541714,  581722,  650849,  832964 , 881392 , 890426,  908610,  914337,  926219, \
                 1037478, 1091764, 1115995, 1141312, 1167122 ,1167208, 1211324, 1211861, 1211987, \
                 1212231, 1212743, 1212872, 1213129, 1306814 ,1306865 ,1315393 ,1369944, 1444632, \
                 1468867, 1468923 ,1560030 ,1631446 ,1670747 ,1778312 ,1797724])
            human_filtered_bounds_90 = np.delete(human_seg_90, (8, 9,10, 11, 15,16,18,19,20, 14, 22, 23, 26, 29,40, \
                                                      42,43, 44, 45,46,48,49, 54, 35,37, 28, 29, 30, 32, 33,34))
            human_filtered_bounds_90 = human_filtered_bounds_90
            human_filtered_bounds = human_filtered_bounds_90
           
        elif self.story =='Tunnel':
            #print('here')
            human_boundaries=  np.array([ 24,25,26,53, 68,83,84,85,86,87,110, 111, 112, 113, 114, \
                                         142, 143,  163, 164,  165,  166,  210, 211, 212,  213, \
                                         214, 243,  244,  245,   246,  304,  305, \
                                         306, 307, 308 ,419,  420,  421,   422, 454, 455,499, 500, 501, 502, \
                                         503,541,542,543,544, 545,567, 568, 569,654,919,920, 921, 922,923, \
                                         1000, 1001,1002,1003,1007, 1023])
            human_boundaries=human_boundaries*1.5 ## each TR was 1.5 seconds in the Tunnel data
            human_filtered_bounds=np.delete(human_boundaries, (0,2,5,6,8,9,10,11,13,14,15,17, \
                                                           18,20, 21,23,24,25,26,27,29,30, \
                                                           31,32,34,35,36,38,39,41,42,44,45,46,47,\
                                                           48,50,51,53,55,56,58,59,60,61,63))
            human_filtered_bounds*= 1000
        
        return human_filtered_bounds
    
    def load_human_bounds_ms(self):
     
        if self.story == "Pieman":
            fp = open(self.path + self.story + "BoundsRun1.txt",
                      encoding='utf-8-sig')        # load in the boundaries from file
            Bounds1_ms = np.array(list(fp.read().splitlines()), dtype=int)

            fp = open(self.path + self.story +"BoundsRun2.txt",
                      encoding='utf-8-sig')        # load in the boundaries from file
            Bounds2_ms = np.array(list(fp.read().splitlines()), dtype=int)
        else:
            fp = open(self.path + self.story + "Bounds.txt",
                      encoding='utf-8-sig')        # load in the boundaries from file
            Bounds1_ms = np.array(list(fp.read().splitlines()), dtype=int)
            Bounds2_ms = None
        
        return Bounds1_ms, Bounds2_ms
   
    def load_button_presses(self):
        if self.story == "Pieman":
            
            bounds1 = sp.io.loadmat(
                self.path + "alldata_run1.mat",)['alldata_run1']
            bounds2 = sp.io.loadmat(
                self.path + "alldata_run2.mat",)['alldata_run2']
            return bounds1, bounds2, None, None
           
        elif self.story == "Tunnel":
            bounds1 = sp.io.loadmat(
                  self.path + "alldata.mat",)['alldata']
            return bounds1, None, None, None
        elif self.story == "Monkey":
            bounds1 = np.load(
                self.path + "monkey_press_time1.npy",)
            bounds2 = np.load(
                self.path + "monkey_press_time2.npy",)
            bounds3 = np.load(
                self.path + "monkey_press_time3.npy",)
            bounds4 = np.load(
                self.path + "monkey_press_time4.npy",)
            return (bounds1[:444455,].T, 
                    bounds2[:415936,].T, 
                    bounds3[:482353,].T, 
                    bounds4[:462256,].T)
        else:
            return None, None, None, None
            
  
      
   # def get_individual_bounds:
  
        
class Output_Loader:
    def __init__(self, story, version, n_iter):
        self.story = story
        self.version = version
        self.n_iter = n_iter
        self.path = os.pardir + '/' + story +  '/outputs/' 

        if story == "Monkey":
            print(story)
        elif story == "Tunnel":
            print(story)

        elif story == "Pieman":
            print(story)
        else: 
            print('invalid story name')      
            
    def load_bound_vec(self, iteration):
        with open(self.path + self.story  + 
                                 '_iter_' + str(iteration) +  
                                 '_version_' + self.version[:-1] +
                                 '_Boundary_Vector' +'.pkl', 'rb') as f:
            return pickle.load(f)
    def load_event_list(self, iteration):
        with open(self.path + self.story  + 
                                 '_iter_' + str(iteration) +  
                                 '_version_' + self.version[:-1] +
                                 '_Events' +'.pkl', 'rb') as f:
            return pickle.load(f)
    def load_response_list(self, iteration):
        with open(self.path + self.story  + 
                                 '_iter_' + str(iteration) +  
                                 '_version_' + self.version[:-1] +
                                 '_Responses' +'.pkl', 'rb') as f:
            return pickle.load(f)

class Word_Loader:
    def __init__(self, story, text):
        self.story = story
        self.path = os.pardir + '/' + story +  '/sourcedata/' 
        self.text = text
    def load_words_and_times(self):    
        #% read in the words and word on-/offsets and parse them 
        # pieman words are called words_b, not all tunnel words are aligned
        # tunnel needs to be extensively fixed
        
        # load the words
        if self.story == "Pieman":
            words = sp.io.loadmat(self.path + "words.mat")['words_b']
        else:
            words = sp.io.loadmat(self.path + "words.mat")['words']
        
        # somehow this is an array inside of an array
        words_new = [];
        for i in range(len(words)):
            words_new.append(words[i][0])
        words = []
        for i in range(len(words_new)):
            words.append(str(words_new[i][0]))
            
        # load the word onsets and offsets
        w_onsets  = sp.io.loadmat(self.path + "word_onsets.mat")
        w_offsets = sp.io.loadmat(self.path + "word_offsets.mat")
        if self.story == "Pieman":
            w_onsets  = w_onsets['onsets_b']
            w_offsets = w_offsets['offsets_b']
        else:
            w_onsets  = w_onsets['onsets']
            w_offsets = w_offsets['offsets']
        #%
        # find problems in the story:
            #First tokenize the text into words
        words_compare = word_tokenize(
            re.sub(r"[^a-zA-Z0-9 ]", "", ''.join(self.text)))
        # fix wanna gonna that are tokenized into 2 words
        for idx, w in enumerate(words_compare):
            if w.lower() == "wan":
                words_compare[idx] = "wanna"
            elif w.lower() == "gon": 
                words_compare[idx] = "gonna"
        while 'na' in words_compare: 
            words_compare.remove('na')
        
        #%% run this to fix the aligned words (there are missing words in tunnel). Only allow for one word at the time to be skipped!
        index = 0 
        # if a fix has already been done, we don't allow more skips!
        one_fix = True
        for w in words_compare:
            if not(w.lower().replace("'", "") == words[index].lower().replace("'", "")):
            #    print(index)
             #   print(words[index])
                if one_fix == False:
                    break
                # try to continue by inserting the missin word in the word list...
                tmp = words[0:index]
                tmp.append(w)
                tmp  = tmp + words[index:]
                words = tmp
                # and insert a nan into the onset..
                tmp = np.concatenate(
                    (w_onsets[0:index,], np.nan*np.empty(
                        (1,1)), w_onsets[index:,]), axis = 0 )
                w_onsets = tmp
                # ... and offset list
                tmp = np.concatenate(
                    (w_offsets[0:index,], np.nan*np.empty(
                        (1,1)), w_offsets[index:,]), axis = 0 )
                w_offsets = tmp
                one_fix = False
            else: 
                # if nothing had to be fixed on this iteration, we can do another fix later
                one_fix = True
                #break
            index = index + 1
        # now there should be the same number of words in the story and in the 
        # word list with corresponding onsets.
        assert(len(words_compare) == len(words))
        # return the words in the story and the onsets
        return words, w_onsets, w_offsets

    
    