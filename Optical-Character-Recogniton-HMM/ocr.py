#!/usr/bin/python
# Optical Character Recognition using HMM:

'''
HMM is used to design the OCR, where the observed images are used to identify the actual character(which is hidden)
HMM is implemented in three ways:
- Simplified Model : with no transiton from one state to another
- Variable Elimination : Using Forward backward algorithm
- Viterby Decoding : Finding the most probable sequence of characters

The probabilities were calculated as:
- Prior Probability / Initial Distribution : The probability of a character occurring in the training file. Separate probabilities were 
  calculated for upper case letters, lower case letters and special characters
- Emission Probability : Each pixel of the test and train images are compared to calculate the emission probabilities. The probability of an
  observed value being a certain character is (100-noise)^matched_pixel_count x (noise)^unmatched_pixel_count. The noise percentage was set 
  empirically based on performance measures calculated on different test images 
- Transition Probability : The probability of transition from one character to another was calculated from the training text file. The sentences
  were all converted to lower case, upper case and title case to enable the moel to predict any type of input given
'''
#
# ./ocr.py : Perform optical character recognition, usage:
# ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (Shahidhya Ramachandran, Sowmya Ravi, Sriram Sitharaman)
# (based on skeleton code by D. Crandall, Oct 2017)
#

#Import necessary packages:

from PIL import Image, ImageDraw, ImageFont
import sys
import os
import math
import string 

# Defining the global varibales :

global char_set
global CHARACTER_WIDTH
global CHARACTER_HEIGHT
global p
global e
global t
global T
global test_letters
global train_letters

#Defining the character set and the dimensions of each character:

char_set = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25

#Importing the test/train image and text:

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]

# Convert images to sequence of strings:

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result.append([([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT)])   
    return result

# Probability of a character at the beginning of a sentence:

def initial_distribution(fname):
    start_letter = {}
    for i in char_set:
        start_letter[i] = 1
    train_txt = open(fname, 'r');
    for line in train_txt:
        if line[0] in char_set:
            start_letter[line[0]] += 1
    no_lines =  sum(start_letter.values())
    no_values = len(start_letter)
    for key, value in start_letter.items():
        start_letter[key] = -math.log(value*1.0 / (no_lines+no_values)) 
    return start_letter

# Probability of a character occurring in a sentence:

def priors(fname):
    letter = {}
    for i in char_set:
        letter[i] = 1
    train_txt = open(fname, 'r');
    for line in train_txt:
        for char in line:
            if char in char_set:
                letter[char] += 1
    no_chars =  sum(letter.values())
    no_values = len(letter)
    for key, value in letter.items():
        letter[key] = -math.log(value*1.0 / (no_chars+no_values)) 
    return letter


# Probability of one character occurring after another:

def transition_prob(fname):
    transition = {}
    den = {}
    for i in char_set:
        transition[i] = {}
        for j in char_set:
            transition[i][j] = 1
    train_txt = open(fname, 'r');
    for line in train_txt:
        #lowercase transitions
        line_low = line.lower()
        for i in range(len(line_low)-2):
            if line_low[i] in char_set and line_low[i+1] in char_set:
                transition[line_low[i]][line_low[i+1]] += 1
        #Uppercase transitions
        line_up = line.upper()
        for i in range(len(line_up)-2):
            if line_up[i] in char_set and line_up[i+1] in char_set:
                transition[line_up[i]][line_up[i+1]] += 1
        #titlecase transitions
        line_cap = string.capwords(line)
        for i in range(len(line_cap)-2):
            if line_cap[i] in char_set and line_cap[i+1] in char_set:
                transition[line_cap[i]][line_cap[i+1]] += 1
    for i in char_set:
        den[i] = sum(transition[i][j]+1 for j in char_set) 
    for i in char_set:
        for j in char_set:
            transition[i][j] = -math.log(transition[i][j]*1.0/den[i])
    return transition

# Probability of character given the observed image:

def emission_prob(train_img_fname, test_img_fname):   
    train_letters = load_letters(train_img_fname)
    test_letters = load_letters(test_img_fname)
    #print(test_letters)
    emission = {}
    char1_cnt = 1
    for char1 in test_letters:
        test_char = [item for sublist in char1 for item in sublist] 
        emis_list = {}
        for char2_cnt in range(len(train_letters)):
            log_prob = 0
            train_char = [item for sublist in train_letters[char2_cnt] for item in sublist]
            for pixel_no in range(len(train_char)):
                if train_char[pixel_no] == test_char[pixel_no]:
                    emi_prob = 0.9
                else:
                    emi_prob = 0.1
                log_prob += -math.log(emi_prob) 
            emis_list[char_set[char2_cnt]] = log_prob
        emission[char1_cnt] = emis_list
        char1_cnt += 1
    return emission

# Defining the global variables:

test_letters = load_letters(test_img_fname)
train_letters = load_letters(train_img_fname)
p = priors(train_txt_fname)
indi = initial_distribution(train_txt_fname)
e = emission_prob(train_img_fname, test_img_fname)
t = transition_prob(train_txt_fname)
T = len(test_letters)

# Simple HMM:

def simple_hmm(train_img_fname, train_txt_fname, test_img_fname):
    simple_result = []
    merged = []
    for char in e:
        emi = e[char]
        #print(char, emi,p)
        merged = { k: emi.get(k, 0) + p.get(k, 0) for k in set(emi) & set(p) }
        simple_result.append(min(merged, key=merged.get))
    return "". join(i for i in simple_result)

# HMM with variable elimination:

def ve_hmm(train_img_fname, train_txt_fname, test_img_fname):
    ve_result = []
    for i in range(1,T+1):
        # Creating dictionaries to store alpha, beta, final probabilities:
        alpha_dict = [{} for _ in range(1,i)]
        beta_dict = [{} for _ in range(i+1,T+1)]
        prob_yi_x1xT = {}
        
        # Getting alpha of state before i:
        if i == 1:
            alpha_iminus1 = p#{key : 0 for key in char_set}
        else:
            for n in range(1,i):
                if n == 1:
                    for y2 in char_set:
                        for y1 in char_set:
                            alpha_dict[n-1][y2] = (e[1][y1] + t[y1][y2] + p[y1])
                else:       
                    for y_nplus1 in char_set:
                        for y_n in char_set:
                            alpha_dict[n-1][y_nplus1] = (e[n][y_n] + t[y_n][y_nplus1] + alpha_dict[n-2][y_n])
            alpha_iminus1 = (alpha_dict[-1])
            
        # Getting beta of state after i:
        if i == T:
            beta_iplus1 = {key : 0 for key in char_set}
        else:
            for n in range(T,i,-1):
                if n == T:
                    for y_Tminus1 in char_set:
                        for y_T in char_set:
                            beta_dict[n-i-1][y_Tminus1] = t[y_Tminus1][y_T] + e[n][y_T]
                else:
                    for y_nminus1 in char_set:
                        for y_n in char_set:
                            beta_dict[n-i-1][y_nminus1] = t[y_nminus1][y_n] + e[n][y_n] + beta_dict[n-i][y_n] 
            beta_iplus1 = beta_dict[0]
            
        # Getting final probability of each character:
        for y_i in char_set:
            prob_yi_x1xT[y_i] = e[i][y_i] + alpha_iminus1[y_i] + beta_iplus1[y_i]
        ve_result.append(min(prob_yi_x1xT, key = prob_yi_x1xT.get))
        
    return "". join(i for i in ve_result)

# HMM using Viterbi :

def vit_hmm(train_img_fname, train_txt_fname, test_img_fname):
    fringe_s1 = {}
    for state in range(1,T+1):
        if state == 1:
            for y in char_set:
                fringe_s1[y] = [p[y] + e[1][y] , y]
        else:
            fringe_s2 = {}
            for char1 in char_set:
                temp_fringe = {}
                for char2 in char_set:
                    temp_fringe[char2] = fringe_s1[char2][0] + e[state][char1] + t[char2][char1]
                least_char = min(temp_fringe, key = temp_fringe.get) 
                fringe_s2[char1] = [temp_fringe[least_char], fringe_s1[least_char][1]+char1]
            fringe_s1 = fringe_s2
    vit_result = least_char = min(fringe_s2.values())[1] 
    return vit_result


simple = simple_hmm(train_img_fname, train_txt_fname, test_img_fname)
print("Simple: " + simple)
ve = ve_hmm(train_img_fname, train_txt_fname, test_img_fname)
print("HMM VE: " + ve)
vit = vit_hmm(train_img_fname, train_txt_fname, test_img_fname)
print("HMM MAP: " + vit)
