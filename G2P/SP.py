from config_models import config_sp
from transformer import TransformerBlock
from sp_tokenizer import Tokenizer_sp
import torch


class Stress_Pred:
    def __init__(self,
                 model,
                 tokenizer):

        self.SP = model
        self.tokenizer = tokenizer

        self.SP.eval()

    def __call__(self, srs):
        with torch.no_grad():
            enc_input_tokens = self.tokenizer.encode(srs)
            pad_id = torch.tensor(self.tokenizer.pad_idx)
            enc_num_padding_tokens = 32 - len(enc_input_tokens)
            encoder_input = torch.cat(
                [
                    torch.tensor(enc_input_tokens),
                    pad_id.repeat(enc_num_padding_tokens)
                ],
                dim=0)

            encoder_mask = (encoder_input != pad_id).unsqueeze(0).unsqueeze(0).int()
            label = self.greedy_decode_stress(
                src=encoder_input,
                src_mask=encoder_mask,
                start_token=self.tokenizer.sos_idx,
            )
        return label

    def greedy_decode_stress(self,
                             src,
                             src_mask,
                             start_token):
        len_src = (src != 3).int().sum().item()
        index_vowels = torch.tensor([(idx) for (idx, i) in enumerate(src) if not (i in list_tokens_without_stress)])[:len_src]
        src = src.unsqueeze(0)
        src_mask = src_mask.unsqueeze(0)
        input_decoder = self.SP.encode(src, src_mask)
        label = torch.tensor([]).type_as(src.data)

        for idx in range(len_src):
            if idx in index_vowels:
                label = torch.cat([label, torch.ones(1, 1).type_as(src.data).fill_(src[0][idx])], dim=1)
            else:
                tgt_mask = (torch.tril(torch.ones((label.size(1), label.size(1)))).type_as(src.data)).unsqueeze(0)
                out = self.SP.decode(input_decoder, src_mask, label, tgt_mask)
                prob = self.SP.fc_out(out[:, -1])

                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.data[0]
                label = torch.cat([label, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)

        pred = self.tokenizer.decode(label[0].tolist())[1:-1]
        return pred


tokenizer_sp = Tokenizer_sp(dict_path='D:\APython\G2P_en_lexicon\my_tokenizer\my_dict_256.json')

set_tokens_without_stress = set()
for token, phoneme in tokenizer_sp.idx2token.items():
    if phoneme[-1].isdigit():
        set_tokens_without_stress.add(tokenizer_sp.token2idx[phoneme[:-1]])
list_tokens_without_stress = list(set_tokens_without_stress)

sp_model = TransformerBlock(config=config_sp,
                            tokenizer=tokenizer_sp)
sp_model.load_state_dict(
    torch.load('D:\APython\G2P_en_lexicon\models\model_0.159.pt', map_location=torch.device('cpu')))

SP = Stress_Pred(model=sp_model,
                 tokenizer=tokenizer_sp)

if __name__ == '__main__':
    print(SP(['N', 'IH', 'K', 'IY', 'T', 'AH']))
