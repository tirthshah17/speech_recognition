import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


#audio import settings 

DEFAULT_SAMPLE_RATE = 16000
MAX_INPUT_CHANNELS = 1
WAVE_OUTPUT_FILE = os.path.join(ROOT_DIR,"input_audio.wav")
INPUT_DEVICE = 0
CHUNK_SIZE = 1024