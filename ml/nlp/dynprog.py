##################################################################################################################################################
##This algorithm applies dynamic programming (solving a complex problem one small subtask at a time) to finding the minimum edit distance in NLP##
##################################################################################################################################################


#D[i-1, j] + 1 (delete)
#D[i, j-1] + 1 (insert)
#D[i-1, j-1] + 2 (replace)
#take minimum of these three and fill D[i, j] and iterate further
#if replace, then if src[i] = tar[j] (e.g. both letters are 'y') then add 0, or in other words, don't add the usual replace cost of 2

import numpy as np

def mineddist(word1, word2):

    #initialize a zeros matrix with the dimensions of word1 and word2 as rows/columns dim, add 1 to each for empty placeholder
    distmat = np.zeros((len(word1)+1, len(word2)+1))
    #intialize (hardcode) the first row and column (i.e. from placeholder to first letters)
    distmat[0,0] = 0 #distance from empty placeholder to empty placeholder is 0
    for i in range(len(word1)+1):
        distmat[i,0] = i
    for j in range(len(word2)+1):
        distmat[0,j] = j
    for i in range(1,len(word1)+1): #iterate filling in the fields, row by row
        for j in range(1,len(word2)+1):
            dlt = distmat[i-1, j] + 1
            ins = distmat[i, j-1] + 1
            if word1[i-1] == word2[j-1]:
                repl = distmat[i-1, j-1]
            else:
                repl = distmat[i-1, j-1] + 2
            minchoice = min(dlt, ins, repl) #see comments above
            distmat[i,j] = minchoice #see comments above
    print (distmat)
    print ("The minimum edit distance between " + word1 + " and " + word2 + " is: " + str(int(distmat[len(word1), len(word2)])))


mineddist("play", "stay")