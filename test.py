import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("neuspell")
install("jiwer")

import neuspell
neuspell.seq_modeling.downloads.download_pretrained_model("subwordbert-probwordnoise")

from neuspell import BertChecker
checker = BertChecker()
checker.from_pretrained()

subprocess.check_call("apt-get -y install sox ffmpeg")
subprocess.check_call("pip install transformers ffmpeg-python sox")
subprocess.check_call("wget https://raw.githubusercontent.com/harveenchadha/bol/main/demos/colab/record.py")

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
from record import record_audio

def load_model():
    # load pretrained model
    processor = Wav2Vec2Processor.from_pretrained("Harveenchadha/vakyansh-wav2vec2-indian-english-enm-700")
    model = Wav2Vec2ForCTC.from_pretrained("Harveenchadha/vakyansh-wav2vec2-indian-english-enm-700")
    return processor, model

processor, model = load_model()

def parse_transcription(wav_file):
    # load audio    
    audio_input, sample_rate = librosa.load(wav_file, sr=16000)

    # pad input values and return pt tensor
    input_values = processor(audio_input, sampling_rate=16_000, return_tensors="pt").input_values
    #print(input_values)
    #print(input_values.shape)

    # INFERENCE
    # retrieve logits & take argmax
    logits = model(input_values).logits
    #print(logits)
    #print(logits.shape)
    predicted_ids = torch.argmax(logits, dim=-1).squeeze()
    #print(predicted_ids)
    #print(predicted_ids.shape)

    softmax_probs = torch.nn.Softmax(dim=-1)(logits)
    seq_prob = torch.max(softmax_probs, dim=-1).values.squeeze()
    rem_tokens = [0,1,2,3,4]
    #print(sum([(token not in rem_tokens) for token in predicted_ids]))
    seq_prob = seq_prob[[(token not in rem_tokens) for token in predicted_ids]]
    #print(seq_prob)
    #print(seq_prob.shape)
    confidence = (torch.prod(seq_prob)**(1/len(seq_prob))).item()

    # transcribe
    transcription = processor.decode(predicted_ids, skip_special_tokens=True)
    cleaned_transcription = checker.correct(transcription)
    return confidence, cleaned_transcription

passage_list = []
passage_list.append('''Opera refers to a dramatic art form, originating in Europe, in which the emotional content is conveyed to the audience 
as much through music, both vocal and instrumental, as it is through the lyrics.''')

passage_list.append('''Dolphins are regarded as the friendliest creatures in the sea and stories of them helping drowning sailors have been common 
since Roman times. The more we learn about dolphins, the more we realize that their society is more complex than people previously imagined.''')

passage_list.append("""Erosion of America's farmland by wind and water has been a problem since settlers first put the prairies and grasslands under the 
plow in the nineteenth century. By the 1930s, more than 282 million acres of farmland were damaged by erosion.""")

passage_list.append("""Naval architects never claim that a ship is unsinkable, but the sinking of the passenger-and-car ferry Estonia in the Baltic surely 
should have never have happened. It was well designed and carefully maintained.""")

passage_list.append("""Mouse and Lion are friends. They live in the African grasslands. Lion
likes to sleep under a shade tree for most of the day. It is very hot in the
grasslands. Every day, Mouse runs far and wide through the grasslands.
He likes to know everything that is going on. In the evening, Mouse comes
back and tells Lion all the news. After the sun goes down and the air gets
cooler, Lion sometimes decides to go for a run. He lowers his huge head
and invites Mouse to hop up. Mouse hops up, and then hangs on for dear
life""")

passage_list.append("""The littlest dragon on the mountain was called Sparkle, because his
hide was pure white, and sparkly. Some of the dragons were blue, some
were green, and some were red. But Sparkle was the only all-white dragon
in their group""")

passage_list.append("""There is a new water park in town. We
go there on the first day of summer.
It has pools and water slides.
There are sprinklers too. The
slides are scary at first. After the
first ride, we love the water slides.
The sprinklers are cool on hot days. One of the pools
makes its own waves. All the kids try to surf the
waves. It is really fun.""")

passage_list.append("""The sky can be full of water. But most of the time you can't
see the water. The drops of water are too small to see. They
have turned into a gas called water
vapor. As the water vapor goes higher
in the sky, the air gets cooler. The
cooler air causes the water droplets
to start to stick to things like bits of
dust, ice or sea salt.""")

import random
import jiwer

def get_passage():

    ground_truth_passage = random.choice(passage_list)
    print('READ PASSAGE : \n')
    print(ground_truth_passage)

def view_transcription():
    print(transcription)

def get_results():
    
    transformation = jiwer.Compose([
                                jiwer.ToLowerCase(),
                                jiwer.RemovePunctuation(),
                                jiwer.RemoveWhiteSpace(replace_by_space=True),
                                jiwer.RemoveMultipleSpaces(),
                                jiwer.Strip(),
                                jiwer.SentencesToListOfWords(),
                                jiwer.RemoveEmptyStrings()
])

    metrics = jiwer.compute_measures(ground_truth_passage, transcription, truth_transform=transformation, hypothesis_transform=transformation)

    audio_dur = librosa.get_duration(filename='test.wav')
    n_words = len(transcription.split())
    wpm = (60*n_words)/audio_dur

    print('RESULTS :\n')
    print('WORD ERROR RATE : {}%'.format(round(metrics['wer']*100,2)))
    print('WORDS PER MINUTE : {}'.format(round(wpm,0)))
    print('CONFIDENCE : {}'.format(confidence))

