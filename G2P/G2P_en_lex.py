from SP import SP
from G2P import G2P
from data_preparation import preprocess_text
import string
import json
import time

with open('D:\python\G2P_en_lexicon\data\word2phoneme.json') as json_file:
    phoneme2grapheme_dict = json.load(json_file)
class G2P_en_lexicon:
    def __init__(self, g2p, sp , with_stress=True):
        self.with_stress = with_stress
        self.G2P = g2p
        self.SP = sp

    def pred_with_stress(self, seq):
        return self.SP(self.G2P(seq))

    def check_punctuation(self, word):
        return any(char in string.punctuation for char in word)

    def __call__(self, seq_list):
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
                    if self.with_stress:
                        print(f"{word} -- {self.pred_with_stress(word)}")
                        result.extend(self.pred_with_stress(word) + [' '])
                    else:
                        print(f"{word} -- {self.G2P(word)}")
                        result.extend(self.pred_with_stress(word) + [' '])
            else:
                count_from_dict += 1
                result.extend(phonemes_from_dict + [' '])

        print(f"{count_from_dict} -- было взято из json\n"
              f"{count_from_model} -- были взяты из модели")

        return result[:-1] if result[-1] == ' ' else result


G2P_en_lexicon = G2P_en_lexicon(g2p=G2P,
                                sp=SP,
                                with_stress=True)

if __name__ == '__main__':
    while True:
        print('Ввод текса')
        text_ = str(input())
        start_time = time.time()
        preprocess_seq = preprocess_text(text_)
        print(G2P_en_lexicon(preprocess_seq))
        end_time = time.time()
        print(f"{(end_time - start_time)*1000} мc -- за это время программа была выполнена ")