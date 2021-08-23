import pyaudio, wave
from settings import DEFAULT_SAMPLE_RATE, MAX_INPUT_CHANNELS, WAVE_OUTPUT_FILE, INPUT_DEVICE, CHUNK_SIZE


class Sound():

    def __init__(self):
        
        self.format = pyaudio.paInt16
        self.channels = MAX_INPUT_CHANNELS
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.chunk = CHUNK_SIZE
        self.path = WAVE_OUTPUT_FILE
        self.device = INPUT_DEVICE
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.rec = 1

    def record_audio(self):

        self.rec = 1        
        self.audio = pyaudio.PyAudio()
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=self.device
        )
        self.frames = []
        while self.rec == 1:
            data = stream.read(self.chunk)
            self.frames.append(data)
        
        stream.stop_stream()
        stream.close()
        self.audio.terminate()
        self.save()

    def save(self):
        wavFile = wave.open(self.path,'wb')
        wavFile.setnchannels(self.channels)
        wavFile.setsampwidth(self.audio.get_sample_size(self.format))
        wavFile.setframerate(self.sample_rate)
        wavFile.writeframes(b''.join(self.frames))
        wavFile.close()
    
    def stop_recording(self):
        self.rec = 0


sound = Sound()


