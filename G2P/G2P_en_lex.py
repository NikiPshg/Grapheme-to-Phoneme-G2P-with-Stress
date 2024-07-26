from SP import SP
from G2P import G2P
from data_preparation import preprocess_text
import string
import json
import time

with open('D:\APython\G2P_en_lexicon\data\word2phoneme.json') as json_file:
    phoneme2grapheme_dict = json.load(json_file)


class G2P_en_lexicon:
    def __init__(self, g2p, sp):
        self.G2P = g2p
        self.SP = sp

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
                        print(f"{word} -- {pred_stress}")
                        result.extend(pred_stress + [' '])
                    else:
                        pred_without = self.G2P(word)
                        print(f"{word} -- {pred_without}")
                        result.extend(pred_without + [' '])
            else:
                count_from_dict += 1
                result.extend(phonemes_from_dict + [' '])

        print(f"{count_from_dict} -- from json\n"
              f"{count_from_model} -- from model")
        result = result[:-1] if result[-1] == ' ' else result

        if not with_stress:
            return self.cleaan_stress(result)
        return result


G2P_en_lexicon = G2P_en_lexicon(g2p=G2P,
                                sp=SP)

if __name__ == '__main__':
    text = """The dominant sequence transduction models are based on complex recurrent or
            convolutional neural networks that include an encoder and a decoder. The best
            performing models also connect the encoder and decoder through an attention
            mechanism. We propose a new simple network architecture, the Transformer,
            based solely on attention mechanisms, dispensing with recurrence and convolutions
            entirely. Experiments on two machine translation tasks show these models to
            be superior in quality while being more parallelizable and requiring significantly
            less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task,
            improving over the existing best results, including
            ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
            our model establishes a new single-model state-of-the-art BLEU score of 41.8 after
            training for 3.5 days on eight GPUs, a small fraction of the training costs of the
            best models from the literature. We show that the Transformer generalizes well to
            other tasks by applying it successfully to English constituency parsing both with
            large and limited training data."""

    start_time = time.time()
    print(G2P_en_lexicon(text, with_stress=True))
    end_time = time.time()
    print(f"{(end_time - start_time) * 1000} мc -- за это время программа была выполнена ")
