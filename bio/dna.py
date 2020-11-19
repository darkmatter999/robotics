import sys
import numpy as np

#creates a random DNA strain and saves it to a .txt file

def dna(length):
    dna_array = np.zeros(length)
    dna_array=dna_array.astype(str)
    nucleotides = ['A', 'T', 'G', 'C']
    i=0
    while i < length:
        dna_array[i] = nucleotides[np.random.randint(4)]
        i+=1
    np.savetxt('dna.txt', dna_array, fmt=['%s'])

if __name__ == "__main__":
    length = int(sys.argv[1])
    dna(length)


