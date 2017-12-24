###################################
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####
#Results
#                Words correct:     Sentences correct:
#   0. Ground truth:      100.00%              100.00%
#     1. Simplified:       93.90%               47.50%
#         2. HMM VE:       95.28%               56.10%
#        3. HMM MAP:       95.26%               56.40%
####
#
#HMM MODEL
#Priors - Count of a particular tag/Total number of tags in the document (for each document) 
#Transition - Count of a tag given its previous tag/ count of previous tags
#Emission - Count of a word being a particular tag/ count of all words that are that particular tag
# All of the above are implemented as dictionary, the latter two are implemented as dictionary of dictionary
# Add one smoothing is applied on the transition and emission probabilities and if something out of the transition and
# if emission model is encountered, a low probability is returned
#
#
##########Simplified Method 
# Maximized the prior[tag]*emission[word][tag] for each of the possible tags and return the tag with maximum value
#
##########Variable elimination
# Implemented using forward backward algorithm by calculating the forward chain probability, backward chain probability and 
# emission probability for predicting the tag for each word.
# forward chain probability calcualted as alpha in the code for each time state from the first word
# Backward chain probability calcualted as beta in the code for each time state from the last word
# Using the above three, tag that maximizes the alpha*beta*emissison probability is returned for each time state 
#
#
##########Viterbi algorithm
# Iteratively for each of the time state,
# A priority_path heapq containing the best possible paths that maximizied the current time state probability
#for each of the possible tag is maintained. For word k in the sentence, all of the paths in the priority_path would have a  
# length k. Each of those paths end with the all possible tags
# The priority is calcualted using the max of (product of transition*emission*prev_state probability for each time (each word from start)
# and each state(each possible tag))


import random
import math
import heapq as hq
import numpy as np

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    
    def __init__(self):
        self.emissionProb={}
        self.intialProb={}
        self.transitionProb={}
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        logproba=0
        for i in range(len(sentence)):
            if i==0:
                logproba+=np.log(self.intialProb[label[i]])+np.log(self.find_emission_prob(sentence[i],label[i]))
            else:
                logproba+=np.log(self.find_trans_prob(label[i-1],label[i]))+np.log(self.find_emission_prob(sentence[i],label[i]))
        return logproba
    # Do the training!
    #
    def train(self, data):
        initialCount={}
        transitionCount={}
        for row in data:
            posLine=row[1]

            for i in range(len(posLine)-1):
                if posLine[i] not in initialCount:
                    initialCount[posLine[i]]=1
                else:
                    initialCount[posLine[i]]+=1
                if posLine[i] not in transitionCount:
                    transitionCount[posLine[i]]={posLine[i+1]:1}
                else:
                    if posLine[i+1] not in transitionCount[posLine[i]]:
                        transitionCount[posLine[i]][posLine[i+1]]=1 
                    else:
                        transitionCount[posLine[i]][posLine[i+1]]+=1
                pass
        self.intialProb={i:float(1.0*(v+1)/(len(initialCount)+len(data))) for i,v in initialCount.items()}
        transitionTotalCount={i:sum(transitionCount[i].values()) for i,v in transitionCount.items()}
        self.transitionProb={i:{l:float((1.0*k+1)/(len(v)+transitionTotalCount[i])) for l,k in v.items()} for i,v in transitionCount.items()}
        emissionCount={}
        for row in range(len(data)):
            wordArray=data[row][0]
            posArray=data[row][1]
            for j in range(len(wordArray)):
                if posArray[j] not in emissionCount:
                    emissionCount[posArray[j]]={wordArray[j]:1}
                else:
                    if wordArray[j] not in emissionCount[posArray[j]]:
                        emissionCount[posArray[j]][wordArray[j]]=1 
                    else:
                        emissionCount[posArray[j]][wordArray[j]]+=1 
        emissionTotalCount={i:sum(emissionCount[i].values()) for i,v in emissionCount.items()}
        self.emissionProb={i:{l:float((1.0*k+1)/(len(v)+emissionTotalCount[i])) for l,k in v.items()} for i,v in emissionCount.items()}
    def find_emission_prob(self,word,posTag):
        try:
            return self.emissionProb[posTag][word]
        # If word is a verb and not present in the model,to check if other forms of the word are available
        except:
            #if posTag == "verb":
            #    try:
            #        return self.emissionProb[posTag][word[:-2]]
            #    except:
            #        return 5e-7
            #    try:
            #        return self.emissionProb[posTag][word[:-3]]
            #    except:
            #        return 5e-7
            #    try:
            #        return self.emissionProb[posTag][word[:-2]+"ing"]
            #    except:
            #        return 5e-7
            #    try:
            #        return self.emissionProb[posTag][word[:-3]+"ed"]
            #    except:
            #        return 5e-7
            return 5e-7
    # Functions for each algorithm.
    def find_trans_prob(self,prev_tag,curr_tag):
        try:
            return self.transitionProb[prev_tag][curr_tag]
        except:
            return 1e-8
    def simplified(self, sentence):
        tags=self.intialProb.keys()
        tokenProbDict={}
        inputSentenceTokens=sentence
        first=True
        prob={tag:self.intialProb[tag] for tag in tags}
        for token in inputSentenceTokens:
            for tag in tags:
                if token not in tokenProbDict:    
                    try:
                        tokenProbDict[token]={tag:prob[tag]*self.find_emission_prob(token,tag)}
                    except:
                        tokenProbDict[token]={tag:0}
                else:
                    try:
                        tokenProbDict[token][tag]=prob[tag]*self.find_emission_prob(token,tag)
                    except:
                        tokenProbDict[token][tag]=0
            #if first==True:
            #    first=False
            #    prob={tag:1 for tag in tags}
        outputTagset=[]
        for token in inputSentenceTokens:
            if max(tokenProbDict[token].values())==0:
                outputTagset.append('NOUN')
            else:
                outputTagset.append(max(tokenProbDict[token],key=tokenProbDict[token].get))
        return outputTagset

    def hmm_ve(self, sentence):
        tags=list(self.intialProb.keys())
        tokenProbDict={}
        words=sentence
        tokenProbList=[]
        predictedPosTags=[]
        alpha=[]
        if len(words)==1:
            prob=[self.intialProb[tags[j]]*self.find_emission_prob(words[0],tags[j]) for j in range(len(tags))]
            predictedPosTags=[tags[prob.index(max(prob))]]
        else:
            for i in range(len(words)-1):
                if i==0:
                    curr_alpha= [sum([self.intialProb[tags[j]]*self.find_emission_prob(words[i],tags[j])*self.find_trans_prob(tags[j],tag1) for j in range(len(tags))]) for tag1 in tags]
                    alpha.append(curr_alpha)
                else:
                    curr_alpha= [sum([alpha[i-1][j]*self.find_emission_prob(words[i],tags[j])*self.find_trans_prob(tags[j],tag1) for j in range(len(tags))]) for tag1 in tags]
                    alpha.append(curr_alpha)
            beta=[]
            for i in range(len(words)-1,0,-1):
                if i==len(words)-1:
                    curr_beta= [sum([self.find_emission_prob(words[i],tags[j])*self.find_trans_prob(tag1,tags[j]) for j in range(len(tags))]) for tag1 in tags]
                    beta.append(curr_beta)
                else:
                    curr_beta= [sum([beta[len(words)-i-2][j]*self.find_emission_prob(words[i],tags[j])*self.find_trans_prob(tag1,tags[j]) for j in range(len(tags))]) for tag1 in tags]
                    beta.append(curr_beta)
            for i in range(len(words)):
                if i==0:
                    prob=[self.intialProb[tags[j]]*beta[-(i+1)][j]*self.find_emission_prob(words[i],tags[j]) for j in range(len(tags))]
                elif i==len(words)-1:
                    prob=[alpha[i-1][j]*self.find_emission_prob(words[i],tags[j]) for j in range(len(tags))]
                else:
                    prob=[alpha[i-1][j]*beta[-(i+1)][j]*self.find_emission_prob(words[i],tags[j]) for j in range(len(tags))]
        #for i in range(len(words)):
        #    try:
        #        prob=[alpha[i][j]*beta[-(i+1)][j]*self.find_emission_prob(words[i],tags[j]) for j in range(len(tags))]
        #    except:
        #        prob=[alpha[i][j]*self.find_emission_prob(words[i],tags[j]) for j in range(len(tags))]
                predictedPosTags.append(tags[prob.index(max(prob))])
        #print (sentence,tuple(predictedPosTags))
        return predictedPosTags

    def hmm_viterbi(self, sentence):
        tags=list(self.intialProb.keys())
        tokenProbDict={}
        words=sentence
        tokenProbList=[]
        predictedPosTags=[]
        alpha=[]
        for i in range(len(words)):
            alpha.append([])
            if i==0:
                priority_path=[]
                for tag in tags:
                    now_prob=self.intialProb[tag]*self.find_emission_prob(words[i],tag)
                    hq.heappush(priority_path,(-1*now_prob,tag))
                    alpha[i].append(now_prob)
            else:
                temp=[]
                for tag in tags:
                    best_prob=-1
                    for path in priority_path:
                        prev_tag=path[1].split(" ")
                        prev_prob=path[0]*-1
                        now_prob=prev_prob*self.find_emission_prob(words[i],tag)*self.find_trans_prob(path[1].split(" ")[-1],tag)
                        if now_prob>best_prob:
                            best_prob=now_prob
                            best_path=path[1]+" "+tag
                    hq.heappush(temp,(-1*best_prob,best_path))
                priority_path=temp 
                
        return hq.heappop(priority_path)[1].split(" ")


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print "Unknown algo!"

