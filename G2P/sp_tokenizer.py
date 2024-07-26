import json


class Tokenizer_sp:
    def __init__(self, config: dict = None, srs: bool = True, dict_path=None, text=None):
        if config is None:
            config = {}

        self.sos = config.get('BOS_TOKEN', '<sos>')
        self.eos = config.get('EOS_TOKEN', '<eos>')
        self.unk = config.get('UNK_TOKEN', '<unk>')
        self.pad = config.get('PAD_TOKEN', '<pad>')
        self.tokens = []
        self.srs = srs

        if dict_path:
            self.load_dict_from_file(dict_path)
        elif text:
            self.create_tokenizer(text)
        else:
            raise ValueError("Текстов нет")

    def create_tokenizer(self, texts):
        tokens = []

        for phonemes_list in texts:
            for phoneme in phonemes_list:
                tokens.append(phoneme)

        self.tokens = [self.sos, self.eos, self.unk, self.pad] + list(set(tokens))

        self.token2idx = {token: int(i) for i, token in enumerate(self.tokens)}
        self.idx2token = {int(i): token for i, token in enumerate(self.tokens)}

        self.unk_idx = self.token2idx[self.unk]
        self.sos_idx = self.token2idx[self.sos]
        self.eos_idx = self.token2idx[self.eos]
        self.pad_idx = self.token2idx[self.pad]

    def load_dict_from_file(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)

        self.idx2token = {int(token): idx for token, idx in data.items()}
        self.token2idx = {idx: int(token) for token, idx in self.idx2token.items()}

        self.unk_idx = self.token2idx.get(self.unk)
        self.sos_idx = self.token2idx.get(self.sos)
        self.eos_idx = self.token2idx.get(self.eos)
        self.pad_idx = self.token2idx.get(self.pad)

    def tokenize(self, text):
        if not self.srs:
            tokens = []
            for tok in text:
                if tok in self.token2idx:
                    tokens.append(tok)
                else:
                    tokens.append(self.unk_idx)
            return [self.sos] + tokens + [self.eos]
        else:
            return [self.sos] + list(text) + [self.eos]

    def convert_tokens_to_idx(self, tokens):
        idx_list = [self.token2idx.get(tok, self.unk_idx) for tok in tokens]
        return idx_list

    def encode(self, text, seq_len=None):
        tokens = self.tokenize(text)[:seq_len]
        return self.convert_tokens_to_idx(tokens)

    def decode(self, idx_list):
        ans = []
        for idx in idx_list:
            try:
                ans.append(self.idx2token[int(idx)])
            except KeyError:
                ans.append(self.idx2token[self.unk_idx])
        return ans

    def get_vocab_size(self):
        return len(self.token2idx)


if __name__ == "__main__":
    tokenizer_sp = Tokenizer_sp(dict_path='D:\APython\G2P_en_lex\my_tokenizer\my_dict_256.json')
    print(tokenizer_sp.idx2token)
