import itertools
import numpy as np

pos_samples = ['I have great news for you', 'So good to hear that', 'This was an amazing party', 'I enjoyed it very much', 'The birthday was so fun']
neg_samples = ['I have a bad feeling', 'He had no idea what to do next', 'So sorry for you', 'I feel so sorry right now', 'He is a sad guy']

pos_vocabulary = []
for sample in pos_samples:
    pos_vocabulary.append(sample.split()) #tokenize
pos_vocabulary = list(itertools.chain.from_iterable(pos_vocabulary)) #flatten list of 'positive' words

neg_vocabulary = []
for sample in neg_samples:
    neg_vocabulary.append(sample.split()) #tokenize
neg_vocabulary = list(itertools.chain.from_iterable(neg_vocabulary)) #flatten list of 'negative' words

pos_vocabulary = [word.lower() for word in pos_vocabulary] #convert all positive words to lowercase
neg_vocabulary = [word.lower() for word in neg_vocabulary] #convert all negative words to lowercase

lambda_dict = {}

total_pos_occur = len(pos_vocabulary) #extract the total number of words (incl. repeats) of the positive vocabulary corpus
total_neg_occur = len(neg_vocabulary) #extract the total number of words (incl. repeats) of the negative vocabulary corpus

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

complete_vocabulary = []
for word in neg_vocabulary:
    complete_vocabulary.append(word)
for word in pos_vocabulary:
    complete_vocabulary.append(word)

unique_words = len(set(complete_vocabulary)) #the unique words in the whole corpus (i.e. positive + negative) are needed for Laplacian Smoothing in the next step

for entry in lambda_dict:
    lambda_dict[entry][0] = float((lambda_dict[entry][0]+1)/(total_pos_occur+unique_words)) #compute the conditional probabilities incl. Laplacian Smoothing
    lambda_dict[entry][1] = float((lambda_dict[entry][1]+1)/(total_neg_occur+unique_words)) #see above, only for 'negative' entries
    lambda_dict[entry][2] = float(np.log(lambda_dict[entry][0]/lambda_dict[entry][1])) #compute the lambda (log likelihood) values for each entry

#evaluate new example
def classify_new(text):
    score = 0 #initialize final prediction score -- if score is > 0, the example is classfied as positive, otherwise as negative (<0) or neutral (0)
    for word in text.split(' '):
        if word in lambda_dict:
            score+=lambda_dict[word][2]
    if score > 0:
        return (str(score) + ' ' + 'The evaluated sentiment is positive.')
    elif score == 0:
        return ('The evaluated sentiment is neutral.')
    else:
        return (str(score) + ' ' + 'The evaluated sentiment is negative.')

print (classify_new('bad sad bad'))

#to do: implement lowercase for testing, implement logprior