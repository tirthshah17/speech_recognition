import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("neuspell")
install("jiwer")

import neuspell
neuspell.seq_modeling.downloads.download_pretrained_model("subwordbert-probwordnoise")


subprocess.check_call(["apt-get","-y","install","sox","ffmpeg"])
subprocess.check_call([sys.executable,"-m","pip","install","transformers","ffmpeg-python","sox"])
subprocess.check_call(["wget","https://raw.githubusercontent.com/harveenchadha/bol/main/demos/colab/record.py"])

