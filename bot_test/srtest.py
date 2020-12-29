import speech_recognition as sr
import pyaudio
import time
import deepspeech
import wave
import numpy as np

mic = sr.Microphone()

r = sr.Recognizer()
#r.energy_threshold = 4000


def record():
    with mic as source:
        #r.adjust_for_ambient_noise(source, duration = 1)
        r.energy_threshold = 12000
        aud = r.listen(source)
        with open('speech.wav', 'wb') as f:
            f.write(aud.get_wav_data(16000))

def infer(fn):
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

record()
message = infer('speech.wav')
print (message)