#########################################################################################################################################
#Quick implementation of Naive Bayes (using conditional probabilities to infer a sentiment), no use is made of stemming and punctuation.#
#########################################################################################################################################

import itertools
import numpy as np

#Setting up a very tiny training set, consisting here of 5 'positive' and 6 'negative' text snippets
pos_samples = ['I have great news for you', 'So good to hear that', 'This was an amazing party', 'I enjoyed it very much', 'The birthday was so fun']
neg_samples = ['I have a bad feeling', 'He had no idea what to do next', 'So sorry for you', 'I feel so sorry right now', 'He is a sad guy', 'A pity']

pos_vocabulary = []
for sample in pos_samples:
    pos_vocabulary.append(sample.split()) #tokenize
pos_vocabulary = list(itertools.chain.from_iterable(pos_vocabulary)) #flatten list of 'positive' words using itertools.chain

neg_vocabulary = []
for sample in neg_samples:
    neg_vocabulary.append(sample.split()) #tokenize
neg_vocabulary = list(itertools.chain.from_iterable(neg_vocabulary)) #flatten list of 'negative' words using itertools.chain

pos_vocabulary = [word.lower() for word in pos_vocabulary] #convert all positive words to lowercase
neg_vocabulary = [word.lower() for word in neg_vocabulary] #convert all negative words to lowercase

lambda_dict = {} #initialize a dictionary in which the lambda (log likelihood) values for each word will be stored

total_pos_occur = len(pos_vocabulary) #extract the total number of words (incl. repeats) of the positive vocabulary corpus
total_neg_occur = len(neg_vocabulary) #extract the total number of words (incl. repeats) of the negative vocabulary corpus

#for both positive and negative vocab corpuses, add the respective occurrences to the first (pos.) or second (neg.) column of a list which
#is the dictionary value
for word in pos_vocabulary:
    if word not in lambda_dict:
        lambda_dict[word] = [1,0,0]
    else:
        lambda_dict[word][0]+=1

for word in neg_vocabulary:
    if word not in lambda_dict:
        lambda_dict[word] = [0,1,0]
    else:
        lambda_dict[word][1]+=1

complete_vocabulary = pos_vocabulary + neg_vocabulary #combine positive and negative vocab corpuses to be able to extract unique words at the next step.

unique_words = len(set(complete_vocabulary)) #the unique words in the whole corpus (i.e. positive + negative) are needed for Laplace Smoothing in the next step

for entry in lambda_dict:
    lambda_dict[entry][0] = float((lambda_dict[entry][0]+1)/(total_pos_occur+unique_words)) #compute the conditional probabilities incl. Laplace Smoothing
    lambda_dict[entry][1] = float((lambda_dict[entry][1]+1)/(total_neg_occur+unique_words)) #see above, only for 'negative' entries
    lambda_dict[entry][2] = float(np.log(lambda_dict[entry][0]/lambda_dict[entry][1])) #compute the lambda (log likelihood) values for each entry

#evaluate new example
def classify_new(text):
    #initialize final prediction score with the logprior which is the ratio between positive and negative test examples.
    #The logprior is used to offset a biased effect when one class has more examples than the other. It is 0 (log of 1) if positive and
    #negative training examples are balanced
    score = np.log(len(pos_samples)/len(neg_samples)) 
    for word in text.split(' '):
        word = word.lower() #convert words to lowercase if applicable
        if word in lambda_dict:
            score+=lambda_dict[word][2] #add the respective lambda value (in third column of the lambda_dict values) to the score
    if score > 0: #evaluate text as positive if score > 0
        return ('The evaluated sentiment is positive. The score is ' + str(score) + '.')
    elif score == 0: #evaluate as neutral if the score is exactly 0.
        return ('The evaluated sentiment is neutral. The score is ' + str(score) + '.')
    else: #evaluate text as negative if score < 0
        return ('The evaluated sentiment is negative. The score is ' + str(score) + '.')

print (classify_new('sad SAD sad sad GOOD gReAt bad BAD BaD bAd good good great amazing ENJOYED good'))

#Here, sentiment analysis was being used as an example of a Naive Bayes application.
#Other possible applications include spam filtering, word disambiguation, author classfication or information relevance validation.