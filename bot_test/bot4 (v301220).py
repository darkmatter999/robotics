#************************************************************************************************************
#***************************AIML/DEEP LEARNING HYBRID VOICE CHATBOT V0.01************************************
#************************************************************************************************************

#This code has been tested on Ubuntu 20.04 and Windows 10.
#In Windows, it apparently does not work with the venv because it does not accept the pre-trained DeepSpeech model
#Furthermore, to get pyaudio to work in the Win10 environment, it is necessary to move the libdeepspeech.so file from
#lib/ to root/
#As of now, pyttsx3 does not work in Python 3.8. It has been successfully tested in Python 3.7.9, though.

#Future challenges:
#implement a true audio stream (i.e. getting rid of saving the recording first and then post-process it)
#implement AIML 2.0 instead of 1.0
#post-process conversational elements

import aiml
import deepspeech #Mozilla Speech Recognition Framework
import wave #Module for processing .wav audio
import numpy as np
import pyaudio #Audio processing library
import time #for (optional) timing
import pyttsx3 #Library for TTS
from scipy.io import wavfile
import struct
import os
import sys
import math


#initialize two empty lists for keeping track of messages and responses
response_list = []
message_list = []

#Since DeepSpeech audio recording streaming doesn't currently work, below is a naive 'fake audio streamer'.
#The function records an audio snippet via PyAudio and saves it to 'output.wav' for further processing via
#DeepSpeech and AIML
def record_question():
    Threshold = 30

    SHORT_NORMALIZE = (1.0/32768.0)
    chunk = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    swidth = 2

    TIMEOUT_LENGTH = 2

    #f_name_directory = r'C:\Users\oliver'

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
            #print('Noise detected, recording beginning')
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
            #n_files = len(os.listdir(f_name_directory))

            #filename = os.path.join(f_name_directory, '{}.wav'.format(n_files))
            filename = 'output.wav'
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(recording)
            wf.close()
            #print('Written to file: {}'.format(filename))
            #print('Returning to listening')



        def listen(self):
            print('Listening beginning')
            listen = True
            while listen == True:
                input = self.stream.read(chunk)
                rms_val = self.rms(input)
                if rms_val > Threshold:
                    self.record()
                    listen = False

    a = Recorder()

    a.listen()

def record_wakeup():
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1 #IMPORTANT: Don't play with this, channels has to be set to 1
    fs = 16000  # Record at 16000 samples per second, has to be compatible with DeepSpeech
    seconds = 2 #length of recording, how can this be made more flexible?
    filename = "wakeup.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    #print('Waiting for initiation')

    stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for n seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    #print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    #time.sleep(5) #optional, for better sync. But proven to be unnecessary at this point.


#do STT with DeepSpeech and the (very time-intensive) main trained model from Mozilla
def audio(fn):
    model_filepath = 'deepspeech-0.9.3-models.tflite'
    model = deepspeech.Model(model_filepath)

    #scorer_filepath = 'deepspeech-0.9.3-models.scorer'
    #model.enableExternalScorer(scorer_filepath)
    #lm_alpha = 0.75
    #lm_beta = 1.85
    #model.setScorerAlphaBeta(lm_alpha, lm_beta)

    filename = fn
    w = wave.open(filename, 'r')
    rate = w.getframerate()
    frames = w.getnframes()
    buffer = w.readframes(frames)

    data16 = np.frombuffer(buffer, dtype=np.int16)

    text = model.stt(data16)
    return (text)


# Create the kernel and learn AIML files
kernel = aiml.Kernel()
kernel.learn("std-startup.xml")
kernel.respond("load aiml b")

#We toggle between 'waiting for initiation' (i.e. hotword utterance) and 'talk mode'
#In practice, the system should work more 'Alexa-like', i.e. 'always-on' and always listening for the hotword before the actual 
#conversational content or question.

first_answers = ['yes', 'how can i help you', 'here for you', 'ready', 'at your service']

def initiation():
    engine = pyttsx3.init()
    wakeup_wait = True
    while wakeup_wait:
        record_wakeup() #wait for 'talk' hotword
        recorded = 'wakeup.wav'
        data = wavfile.read(recorded)
        if data[1].max() > 2000:
            message = audio('wakeup.wav')
            print (message)
            if message == "hello":
                #make a 'silent' response so that certain AIML attributes can be loaded
                #(see basic_chat.aiml script)
                bot_response = kernel.respond(message) 
                time.sleep(1)
                engine.say(first_answers[np.random.randint(5)])
                engine.runAndWait()
                wakeup_wait = False
                conversation() #if hotword is heard, switch to open conversation mode




def conversation():
    while True:
        record_question()
        message = audio('output.wav')
        #message = message.split(' ')[:1]
        #message = ' '.join(message)
        print (message)
        if message == "hello exit":
            initiation()
        elif message == "hello save":
            kernel.saveBrain("bot_brain.brn")
        elif message == "hello type":
            message = input("Enter your message to the bot: ")
            bot_response = kernel.respond(message)
            print (bot_response)
        elif message == "":
            #engine = pyttsx3.init()
            #engine.say('You said nothing')
            #engine.runAndWait()
            pass
        else:
            bot_response = kernel.respond(message)
            #message_list.append(message)
            #response_list.append(bot_response)
            # Do something with bot_response
            #print (bot_response)
            #time.sleep(3)
            engine = pyttsx3.init() #output the bot's reponse via pyttsx3 (TTS)
            engine.say(bot_response)
            engine.runAndWait()
    
initiation()
#conversation()

#good morning 1 1
#can you go with me 1 1
#happy birthday 0 0
#what can you teach me 1 1
#you are bad 1 0
#that is my name 1 1
#seriously 1 1
#show me a picture of 0 0
#teach me how to read 1 1
#do you smoke 1 0

#14/20 (70%) points reached in field test Oliver/Nicolina on 30/12/2020
#The first number on the right of the test expression refers to whether the system understood what Oliver said:
#1 for understood, 0 for not understood. Correspondingly, the second number refers to Nicolina's input.

#Here, we tried with multiple AIML <think> tags in one template, so predefining some attributes.
#On initiation (hotword utterance) the AIML is handled and these attributes are loaded into the system.
#Ideas: 
#Overcome latency: 'Eye contact' as threshold for speaking
#Overcome signal strength/threshold issues: implement flexible ambient noise meter
#Overcome hard-coded recording duration: measure microphone level also while recording and shut
#recording off upon silence/signal strength falling below threshold