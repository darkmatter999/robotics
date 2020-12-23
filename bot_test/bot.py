#************************************************************************************************************
#***************************AIML/DEEP LEARNING HYBRID VOICE CHATBOT V0.01************************************
#************************************************************************************************************

#This code has been tested on Ubuntu 20.04 and Windows 10.
#In Windows, it apparently does not work with the venv because it does not accept the pre-trained DeepSpeech model
#Furthermore, to get pyaudio to work in the Win10 environment, it is necessary to move the libdeepspeech.so file from
#lib/ to root/
#As of now, pyttsx3 does not work in Python3.8. It has been successfully tested in Python3.7.9, though.

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

#Since DeepSpeech audio recording streaming doesn't currently work, below is a naive 'fake audio streamer'.
#The function records an audio snippet via PyAudio and saves it to 'output.wav' for further processing via
#DeepSpeech and AIML
def record_question():
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1 #IMPORTANT: Don't play with this, channels has to be set to 1
    fs = 16000  # Record at 16000 samples per second, has to be compatible with DeepSpeech
    seconds = 3 #length of recording, how can this be made more flexible?
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

    #time.sleep(5) #optional, for better sync. But proven to be unnecessary at this point.

def record_wakeup():
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1 #IMPORTANT: Don't play with this, channels has to be set to 1
    fs = 16000  # Record at 16000 samples per second, has to be compatible with DeepSpeech
    seconds = 3 #length of recording, how can this be made more flexible?
    filename = "wakeup.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Waiting for initiation')

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

    #time.sleep(5) #optional, for better sync. But proven to be unnecessary at this point.


#do STT with DeepSpeech and the (very time-intensive) main trained model from Mozilla
def audio(fn):
    model_filepath = 'deepspeech-0.9.3-models.pbmm'
    model = deepspeech.Model(model_filepath)

    scorer_filepath = 'deepspeech-0.9.3-models.scorer'
    model.enableExternalScorer(scorer_filepath)
    lm_alpha = 0.75
    lm_beta = 1.85
    model.setScorerAlphaBeta(lm_alpha, lm_beta)

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

'''
# Press CTRL-C to break this loop
while True:
    message = input("Enter your message to the bot: ")
    if message == 'audio':
        record() #open recording instance
        #message = audio('audio/8455-210777-0068.wav')
        #It is a 'fake audio stream', i.e. here we have to hardcode the output file
        #In future, this should be replaced by STREAM --> INFERENCE --> OUTPUT
        #Right now, it is RECORDING --> OUTPUT FILE --> INFERENCE --> OUTPUT
        message = audio('output.wav') 
        #print (kernel.respond(message)) #optional display of response
    if message == "quit":
        exit()
    elif message == "save":
        kernel.saveBrain("bot_brain.brn")
    else:
        bot_response = kernel.respond(message)
        # Do something with bot_response
        #print (bot_response)
        engine = pyttsx3.init() #output the bot's reponse via pyttsx3 (TTS)
        engine.say(bot_response)
        engine.runAndWait()
'''

# Alternate version, here starting with voice input. 'Talk' initiates the conversation
# To be improved in the future: 
# 1. What happens with exceptions?
# 2. --- TBD
'''
wakeup_wait = True
talk = False

while wakeup_wait:
    record_wakeup() #wait for 'talk' hotword
    message = audio('wakeup.wav')
    if message == 'talk':
        wakeup_wait = False
        talk = True
        while talk:
            record_question() #open recording instance
            #message = audio('audio/8455-210777-0068.wav')
            #It is a 'fake audio stream', i.e. here we have to hardcode the output file
            #In future, this should be replaced by STREAM --> INFERENCE --> OUTPUT
            #Right now, it is RECORDING --> OUTPUT FILE --> INFERENCE --> OUTPUT
            message = audio('output.wav') 
            #print (kernel.respond(message)) #optional display of response
            if message == "exit":
                #exit()
                
                #break
                
                talk = False
                #continue
            elif message == "save":
                kernel.saveBrain("bot_brain.brn")
            elif message == "type":
                message = input("Enter your message to the bot: ")
                bot_response = kernel.respond(message)
                print (bot_response)
            else:
                bot_response = kernel.respond(message)
                # Do something with bot_response
                print (bot_response)
                time.sleep(3)
                #engine = pyttsx3.init() #output the bot's reponse via pyttsx3 (TTS)
                #engine.say(bot_response)
                #engine.runAndWait()
    if message == 'exit':
        print (wakeup_wait)
        exit()
    '''

#We toggle between 'waiting for initiation' (i.e. hotword utterance) and 'talk mode'
#In practice, the system should work more 'Alexa-like', i.e. 'always-on' and always listening for the hotword before the actual 
#conversational content or question.

def initiation():
    wakeup_wait = True
    while wakeup_wait:
        record_wakeup() #wait for 'talk' hotword
        message = audio('wakeup.wav')
        if message == 'talk':
            wakeup_wait = False
            conversation() #if hotword is heard, switch to open conversation mode
        elif message == 'exit':
            exit()

def conversation():
    talk = True
    while talk:
        record_question()
        message = audio('output.wav')
        if message == "exit":
            talk = False
            initiation() #if 'exit' is heard, switch back to hotword listening, or initiation
        elif message == "save":
            kernel.saveBrain("bot_brain.brn")
        elif message == "type":
            message = input("Enter your message to the bot: ")
            bot_response = kernel.respond(message)
            print (bot_response)
        else:
            bot_response = kernel.respond(message)
            # Do something with bot_response
            #print (bot_response)
            time.sleep(3)
            engine = pyttsx3.init() #output the bot's reponse via pyttsx3 (TTS)
            engine.say(bot_response)
            engine.runAndWait()
    
initiation()