import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa

import random
import jiwer

from neuspell import BertChecker
checker = BertChecker()
checker.from_pretrained()

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
passage_list.append('''Opera refers to a dramatic art form, originating in Europe,
in which the emotional content is conveyed to the audience as much through music,
both vocal and instrumental, as it is through the lyrics.''')

passage_list.append('''John went for a bike ride. He rode
around the block. Then he met some girls he knew from school.
They all rode to the field to play. John had a great time playing
games with his friends.''')

passage_list.append("""Tim went to the park with his brother. They
brought baseballs and gloves. They played catch for two hours. It started
to get very hot out, so they went home for some lemonade. They
had a great day.""")

passage_list.append("""The ocean has bright blue water
filled with waves. Many types of fish live in the ocean.
Seagulls love flying over the ocean to look for fish.
There is soft sand along the shore, and there are pretty seashells
in the sand. The ocean is a great place to visit.""")

passage_list.append("""The kids were outside playing catch. They heard a
rumble in the sky. They didn’t want to stop
playing, but they knew it wasn’t safe to be out in
a storm. Also, they did not want to get wet.
They decided to go inside and play a board game.
They loved listening to the thunder as they played their game.
The kids went outside again after the storm had
passed. They saw a rainbow!""")



def get_passage():
    
    ground_truth_passage = random.choice(passage_list)
    print('READ PASSAGE : \n')
    print(ground_truth_passage)
    return ground_truth_passage

def view_transcription():
    print(transcription)

def get_results(ground_truth_passage, transcription, confidence):
    
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
