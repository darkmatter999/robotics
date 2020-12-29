#************************************************************************************************************
#***************************AIML/DEEP LEARNING HYBRID VOICE CHATBOT V0.01************************************
#************************************************************************************************************

#This code has been tested on Ubuntu 20.04 and Windows 10.
#In Windows, it apparently does not work with the venv because it does not accept the pre-trained DeepSpeech model
#Furthermore, to get pyaudio to work in the Win10 environment, it is necessary to move the libdeepspeech.so file from
#lib/ to root/
#As of now, pyttsx3 does not work in Python 3.8. It has been successfully tested in Python 3.7.9, though.

#Future challenges:
#post-process conversational elements

import aiml
import deepspeech #Mozilla Speech Recognition Framework
import wave #Module for processing .wav audio
import numpy as np
import pyaudio #Audio processing library
import time #for (optional) timing
import pyttsx3 #Library for TTS
from scipy.io import wavfile #a scipy function used to analyze whether a given audio snippet contains an actual audio input


#--- FOR FUTURE USE --- Initialize two empty lists for keeping track of messages and responses
response_list = []
message_list = []

#Open a pyaudio Recording Stream
#The function records an audio snippet via PyAudio and saves it to 'output.wav' for further processing via
#DeepSpeech and AIML
def record_question():
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1 #IMPORTANT: Don't play with this, channels has to be set to 1
    fs = 16000  # Record at 16000 samples per second, has to be compatible with DeepSpeech
    seconds = 5 #hard-coded maximum duration of a user's statement/question. ***Can this be handled more flexibly?***
    filename = "output.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')

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

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

#Recording Stream for hotword detection
def record_wakeup():
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1 #IMPORTANT: Don't play with this, channels has to be set to 1
    fs = 16000  # Record at 16000 samples per second, has to be compatible with DeepSpeech
    #length of recording, just enough for the hotword utterance. If it was longer, there would be less 'input flexibility'
    #(the user would frequently find that in his instant of talking the machine is not ready because her utterance happened
    #in between the recording period. That's why it is important to limit the duration to just the time needed for an average
    #speaker to utter the hotword)
    seconds = 2 
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

    
#conduct STT with DeepSpeech TFlite (lightweight 45MB version of the big .pbmm model)
def audio(fn):
    model_filepath = 'deepspeech-0.9.3-models.tflite'
    model = deepspeech.Model(model_filepath)

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

#We toggle between 'waiting for initiation' (i.e. hotword utterance) and 'conversation mode'
#The system is 'always-on', checking for the pre-defined hotword to be uttered, and once recognized, switches
#to the conversation mode in which the system expects a statement/question, after it gave an initial response ('yes', or
#'how can I help you' etc.)

def initiation():
    engine = pyttsx3.init()
    wakeup_wait = True
    while wakeup_wait: #infinite loop of 2-second-long recordings, waiting for hotword utterance. 
        record_wakeup() 
        recorded = 'wakeup.wav'
        data = wavfile.read(recorded)
        #in order to increase speed and reduce unnecessary latency, we check if the user actually said anything by checking via scipy
        #the signal strength. Only if the signal strength is high enough, we start inference, and only if the inference yields the
        #hotword, we switch to the conversation mode
        if data[1].max() > 2000: 
            message = audio('wakeup.wav')
            print (message)
            if message == "hello":
                time.sleep(1)
                engine.say('yes') #response to hotword utterance. Only after outputting this response, the conversation mode can start
                engine.runAndWait()
                wakeup_wait = False
                conversation() #if hotword is heard, switch to open conversation mode

def conversation():
    record_question()
    message = audio('output.wav')
    if message == "hello exit":
        exit()
    elif message == "hello save":
        kernel.saveBrain("bot_brain.brn")
    elif message == "hello type":
        message = input("Enter your message to the bot: ")
        bot_response = kernel.respond(message)
        print (bot_response)
    elif message == "":
        pass
        initiation() #if nothing is being said, we switch back to waiting for hotword (aka initiation mode)
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
        initiation() #switch back to initiation mode once the system's answer has been output
    
initiation()