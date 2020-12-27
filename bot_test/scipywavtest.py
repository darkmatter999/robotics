
import numpy as np
from scipy.io import wavfile

def show_info(aname, a):
    print ("Array", aname)
    print ("shape:", a.shape)
    print ("dtype:", a.dtype)
    print ("min, max:", a.min(), a.max())


rate, data = wavfile.read('final_output.wav')

show_info("data", data)


'''
import wave

infiles = ["wakeup.wav", "output.wav"]

outfile = "final_output.wav"

data= []

for infile in infiles:

    w = wave.open(infile, 'rb')

    data.append( [w.getparams(), w.readframes(w.getnframes())] )

    w.close()

output = wave.open(outfile, 'wb')

output.setparams(data[0][0])

output.writeframes(data[0][1])

output.writeframes(data[1][1])

output.close()
'''