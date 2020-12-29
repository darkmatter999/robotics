'''
import pyaudio
import math
import struct
import wave
import sys

#Assuming Energy threshold upper than 30 dB
Threshold = 100

SHORT_NORMALIZE = (1.0/32768.0)
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
swidth = 2
Max_Seconds = 10
TimeoutSignal=((RATE / chunk * Max_Seconds) + 2)
silence = True
FileNameTmp = 'output2.wav'
Time=0
all =[]

def GetStream(chunk):
    return stream.read(chunk)
def rms(frame):
    count = len(frame)/swidth
    format = "%dh"%(count)
    # short is 16 bit int
    shorts = struct.unpack( format, frame )

    sum_squares = 0.0
    for sample in shorts:
        n = sample * SHORT_NORMALIZE
        sum_squares += n*n
    # compute the rms 
    rms = math.pow(sum_squares/count,0.5)
    return rms * 1000

def WriteSpeech(WriteData):
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(FileNameTmp, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(WriteData)
    wf.close()

def KeepRecord(TimeoutSignal, LastBlock):
    all.append(LastBlock)
    for i in range(0, TimeoutSignal):
        try:
            data = GetStream(chunk)
        except:
            continue
        #I chage here (new Ident)
        all.append(data)

    print ("end record after timeout")
    data = ''.join(all)
    print ("write to File")
    WriteSpeech(data)
    silence = True
    Time=0
    listen(silence,Time)     

def listen(silence,Time):
    print ("waiting for Speech")
    while silence:
        try:
            input = GetStream(chunk)
        except:
            continue
        rms_value = rms(input)
        if (rms_value > Threshold):
            silence=False
            LastBlock=input
            print ("hello I'm Recording....")
            KeepRecord(TimeoutSignal, LastBlock)
        Time = Time + 1
        if (Time > TimeoutSignal):
            print ("Time Out No Speech Detected")
            sys.exit()

p = pyaudio.PyAudio()

stream = p.open(format = FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input = True,
    output = True,
    frames_per_buffer = chunk)

listen(silence,Time)
'''

import pyaudio
import math
import struct
import wave
import time
import os
import sys

Threshold = 100

SHORT_NORMALIZE = (1.0/32768.0)
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
swidth = 2

TIMEOUT_LENGTH = 5

f_name_directory = r'C:\Users\oliver'

class Recorder:

    @staticmethod
    def rms(frame):
        count = len(frame) / swidth
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=chunk)

    def record(self):
        print('Noise detected, recording beginning')
        rec = []
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH

        while current <= end:

            data = self.stream.read(chunk)
            if self.rms(data) >= Threshold: end = time.time() + TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)
        self.write(b''.join(rec))

    def write(self, recording):
        n_files = len(os.listdir(f_name_directory))

        filename = os.path.join(f_name_directory, '{}.wav'.format(n_files))

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()
        print('Written to file: {}'.format(filename))
        print('Returning to listening')



    def listen(self):
        print('Listening beginning')
        while True:
            input = self.stream.read(chunk)
            rms_val = self.rms(input)
            if rms_val > Threshold:
                self.record()

a = Recorder()

a.listen()