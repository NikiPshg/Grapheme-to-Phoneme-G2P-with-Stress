from G2P_lexicon.G2P import G2P
from G2P_lexicon.SP import SP
from G2P_lexicon.data_preparation import preprocess_text
import string
import json
import time
import os


dirname = os.path.dirname(__file__)
json_path = os.path.join(dirname, "data/word2phoneme.json")

with open(json_path) as json_file:
    phoneme2grapheme_dict = json.load(json_file)


class g2p_en_lexicon:
    def __init__(self):
        self.G2P = G2P
        self.SP = SP

    def cleaan_stress(self, seq: list):
        return [phoneme[:-1] if phoneme[-1].isdigit() else phoneme for phoneme in seq]

    def pred_with_stress(self, seq):
        return self.SP(self.G2P(seq))

    def check_punctuation(self, word):
        return any(char in string.punctuation for char in word)

    def __call__(self, seq, with_stress=True):
        seq_list = preprocess_text(seq)
        result = []
        count_from_dict = 0
        count_from_model = 0
        for word in seq_list:
            phonemes_from_dict = phoneme2grapheme_dict.get(word)
            if phonemes_from_dict is None:
                if self.check_punctuation(word):
                    result.extend([word] + [' '])
                else:
                    count_from_model += 1
                    if with_stress:
                        pred_stress = self.pred_with_stress(word)
                        #print(f"{word} -- {pred_stress}")
                        result.extend(pred_stress + [' '])
                    else:
                        pred_without = self.G2P(word)
                        #print(f"{word} -- {pred_without}")
                        result.extend(pred_without + [' '])
            else:
                count_from_dict += 1
                result.extend(phonemes_from_dict + [' '])

        #print(f"{count_from_dict} -- from json\n"
              #f"{count_from_model} -- from model")
        result = result[:-1] if result[-1] == ' ' else result

        if not with_stress:
            return self.cleaan_stress(result)
        return result


if __name__ == '__main__':
    G2P_en_lexicon = g2p_en_lexicon()
    text = """mtusi is the worst option for a programmer or a student"""
    start_time = time.time()
    print(G2P_en_lexicon(text))
    end_time = time.time()
    print(f"{(end_time - start_time) * 1000} мc -- за это была выполнена ")
