#!/usr/bin/env python
'''
Methodology:
1) Parse the input data to create the bag of words and also  get the counts for the corresponding counts of words for each of the classes present in the training set.
    Before that each tweet was preprocessed by remvoving all the special characters, next line characters. This preprocessing is done for the test file as well
    
2) Once we have the bag of words, we create a dictionary called 'feature_list' that contains the probability P(word|class)
    To compute the probability we have used Laplacian smooting where we compute the probability of as follows
    (frequency of a term in a given class + 1)/(total number of tweets in a given class + Vocabulary size)
    Vocabulary size is the total number of terms in all the tweets in the training file.
    
3) The prior of each of the classes is computed.

4) Then, the test file is read line by line. For each word in the tweet, if it exists in the bag of words, the probability of the word belonging to each  
    of the    classes in the training file is retrieved and added to the dictionary.
    Finally we compute the posterior probability by the following formula:  P(word|class)*P(class)
    
5) In order to augment our naive bayes classifier, we also check if a string in a tweet is a substring of a city name (class). If it is, then we assign the  
    city as the result 
    
'''


import pandas as pd
import operator
import numpy as np
import os
import time
import sys

# Get the names of the training file from command line.
train_file_name = sys.argv[1]
test_file_name = sys.argv[2]
output_file = sys.argv[3]
train_file = open(train_file_name)

# dictionary that contains the cumulative counts of a word for each class
parsed_data = {}

# dictionary for bag of words
bag_of_words = {}

# list for each target in the order of tweet
target = []

# Since stop words are common for english we hard coded the nltk stop words 
stopwords=['a','about','above','after','again','against','all','am','an','and','any','are','arent','as','at','be','because','been','before','being','below','between','both','but','by','cant','cannot','could','couldnt','did','didnt','do','does','doesnt','doing','dont','down','during','each','few','for','from','further','had','hadnt','has','hasnt','have','havent','having','he','hed','hell','hes','her','here','heres','hers','herself','him','himself','his','how','hows','i','id','ill','im','ive','if','in','into','is','isnt','it','its','its','itself','lets','me','more','most','mustnt','my','myself','no','nor','not','of','off','on','once','only','or','other','ought','our','ours','ourselves','out','over','own','same','shant','she','shed','shell','shes','should','shouldnt','so','some','such','than','that','thats','the','their','theirs','them','themselves','then','there','theres','these','they','theyd','theyll','theyre','theyve','this','those','through','to','too','under','until','up','very','was','wasnt','we','wed','well','were','weve','were','werent','what','whats','when','whens','where','wheres','which','while','who','whos','whom','why','whys','with','wont','would','wouldnt','you','youd','youll','youre','youve','your','yours','yourself','yourselves']
stopwords_dict = {i:1 for i in stopwords}
t0 = time.time()
#dictioary to keep track of the number of tweets in which a word occurs.
doc_count = {}

# Read the train file line by line
for line in train_file:
    new_array = line.strip().split(" ")
    new_array2=[]
    seen=[]
    target.append( new_array[0])
    if new_array[0] not in parsed_data:
        parsed_data[new_array[0] ] = {}
    for i in range(1,len(new_array)):
        seen.append( new_array[i])
        if  (new_array[i].isalnum() or (new_array[i].isalnum() == False and len(new_array[i]) >1) ) :
            new_array[i]= ''.join(ch for ch in new_array[i] if ch.isalnum())
            # prerocess the tweet by removing non ascii sequences, special characters and converting to lower
            string =new_array[i].decode('ascii','ignore').encode('ascii','ignore').replace("\r"," ").replace('#','').lower()
            if len(new_array[i]) >1 and string not in stopwords_dict:
                new_array2.append(string)  
                
                # adding string and counts to bag of words
                if string not in bag_of_words:
                    bag_of_words[string]=1
                    doc_count[string]=1
                else:
                    bag_of_words[string] +=1
                    if string not in seen:
                        doc_count[string] += 1
                if string  not in parsed_data[new_array[0]]:
                    parsed_data[new_array[0]].update({string:1})
                else: 
                    parsed_data[new_array[0]][string] +=1
    
train_file.close()    
# prior distribution calculation
import collections
target_count = dict(collections.Counter(target))
max_occuring_class=max(target_count.iteritems(), key=operator.itemgetter(1))[0]
target_prob = {k: float(v)/len(target) for k,v in target_count.items()}

# get unique classes
unique_class = list(set(target))
unique_class_first=[ci.split(",")[0].replace("_","").lower() for ci in unique_class]
unique_class_second=[ci.split(",")[1].replace("_","").lower() for ci in unique_class]
ngram_words_class=[[ci[0:i]  for i in range(5,len(ci)+1) if len(ci[0:i])>=5] for ci in unique_class_first]


feature_list = {}
check =0
subset_bag_20={}
for word in bag_of_words:
    if (bag_of_words[word]>=1) :
        subset_bag_20[word]=1

length = len(subset_bag_20)

# Probability Calculation
check=0
for city,wordlist in parsed_data.items():
    check +=1
    sub_length= sum(wordlist.values())
    for word in subset_bag_20:
        if word in wordlist:
            if check ==1 :
                feature_list[word]= {city:(float(wordlist[word] +1  )/float(sub_length + length) ) }
            else:
                feature_list[word].update({city:(float(wordlist[word] +1  )/float(sub_length + length) ) })
        else:
            if check ==1 :
                feature_list[word]= {city:float(1  )/float(sub_length + length)  }
            else:
                feature_list[word].update({city:(float(1  )/float(sub_length + length) ) })


                
prior_df = pd.DataFrame(target_prob.items(),columns= ["City","prior"]).sort_values("City").reset_index(drop= True)
# Read the test file and remove stopwords
test_file_name = "tweets.test1.txt"
default = 0
correct =0
count = 0
test_file = open(test_file_name)
opfile = open(output_file, 'w')
# Read each line of the test file 
for line in test_file:
    count +=1
    new_array = line.strip().split(" ")
    prob  ={}
    result_check=0
    ngram_class_counts=[0]*len(unique_class)
    
    for i in range(1,len(new_array)):
        if  (new_array[i].isalnum() or (new_array[i].isalnum() == False and len(new_array[i]) >1) ) :
            new_array[i]= ''.join(ch for ch in new_array[i] if ch.isalnum())
            # preprocess the tweet
            string =new_array[i].decode('ascii','ignore').encode('ascii','ignore').replace("\r"," ").replace('#','').lower()
            # retrieve the posterior probabilities for each word in the string.
            if string in subset_bag_20:
                prob[string] = feature_list[string]
            #else if string not in stopwords_dict: 
            # If the city name is in the tweet the assign the city to the tweet
            # if string not in stopwords_dict:  
                # for word in unique_class:
                    # if (string in word.replace("_","").lower().split(",")[0] or word.replace("_","").lower().split(",")[0] in string) and (len(string)>=4):
                        # print string
                        # result_check=1 
                        # result = word
            # Check if the string contains a stubstring of a class then include that in the code. 
            if string not in stopwords_dict:
                for classNo in range(len(unique_class)):
                        if len(string)>=4:
                            for token in ngram_words_class[classNo] :
                                if token in string or string in token:
                                    result_check=1
                                    #print string
                                    ngram_class_counts[classNo]+=1

        result=unique_class[ngram_class_counts.index(max(ngram_class_counts))]
    result_check1=0            
    if result_check==0:
        if bool(prob):
            prob_df = pd.DataFrame(prob)
            result = prior_df["City"][np.argmax( np.sum(np.log(np.array(prob_df)),axis=1) + np.log(np.array(prior_df["prior"])))]
        else:
            default+=1
            result = max_occuring_class
    #print result, " ", new_array[0]

    if result == new_array[0]:
        correct +=1
    opfile.write(result + "  "+ line)
    
opfile.close()
test_file.close()
t1 = time.time()
#print "Ended"
print "accuracy",correct * 100.0/count,"%"
#print "Total Time", t1-t0 

# print top 5 words associated with each geolocation:
a = pd.DataFrame(feature_list)
for i in range(len(target_count)):
    print "The top five words for ",a.index[i]," :"
    print a.loc[a.index[i]].nlargest(n=5)

