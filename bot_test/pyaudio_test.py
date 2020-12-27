import pyaudio
import wave

j=1
while j < 6:
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 1
    WAVE_OUTPUT_FILENAME = "file" + str(j) + ".wav"
 
    audio = pyaudio.PyAudio()
 
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
    print ("recording...")
    frames = []
 
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print ("finished recording")
 
 
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
 
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    j+=1


infiles = ["file1.wav", "file2.wav", "file3.wav", "file4.wav", "file5.wav"]

outfile = "complete_output.wav"

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
