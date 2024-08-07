import torch
from G2P_lexicon.transformer import TransformerBlock
from tokenizers import Tokenizer
from G2P_lexicon.config_models import config_g2p
import os

dirname = os.path.dirname(__file__)


def decode_form_G(tokens: str):
    """
    Converts model output to a readable format.
    Args:
        tokens: 'NĠAH1ĠMĠBĠER0ĠZ'
    Returns:
        ['N', 'AH1', 'M', 'B', 'ER0', 'Z']
    """
    return ''.join(tokens).split('Ġ')


class GraphemeToPhoneme:
    def __init__(self,
                 model,
                 tokenizer):

        self.g2p_model = model
        self.tokenizer = tokenizer

        self.g2p_model.eval()

    def greedy_decode_grapheme(self, model,
                               src,
                               src_mask,
                               max_len,
                               start_token):
        src = src.unsqueeze(0)
        src_mask = src_mask.unsqueeze(0)
        input_decoder = model.encode(src, src_mask)
        label = torch.zeros(1, 1).fill_(start_token).type_as(src.data)

        for _ in range(max_len - 1):
            tgt_mask = (torch.tril(torch.ones((label.size(1), label.size(1)))).type_as(src.data)).unsqueeze(0)
            out = model.decode(input_decoder, src_mask, label, tgt_mask)
            prob = model.fc_out(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            label = torch.cat([label, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            if next_word == self.tokenizer.encode("<eos>").ids[0]:
                break

        pred = decode_form_G(self.tokenizer.decode(label[0].tolist()))
        return pred

    def __call__(self, srs):
        with torch.no_grad():
            enc_input_tokens = self.tokenizer.encode(srs).ids
            pad_id = self.tokenizer.encode("<pad>").ids[0]
            enc_num_padding_tokens = 32 - len(enc_input_tokens) - 2
            encoder_input = torch.cat([
                torch.tensor([self.tokenizer.encode("<bos>").ids[0]]),
                torch.tensor(enc_input_tokens),
                torch.tensor([self.tokenizer.encode("<eos>").ids[0]]),
                torch.tensor([pad_id] * enc_num_padding_tokens)
            ], dim=0)

            encoder_mask = (encoder_input != pad_id).unsqueeze(0).unsqueeze(0).int()
            pred = self.greedy_decode_grapheme(
                model=self.g2p_model,
                src=encoder_input,
                src_mask=encoder_mask,
                max_len=32,
                start_token=self.tokenizer.encode("<bos>").ids[0]
            )
        return pred


dict_path = os.path.join(dirname, "my_tokenizer/bpe_256_cmu.json")
model_path = os.path.join(dirname, "models/model_g2p.pt")


tokenizer_g2p = Tokenizer.from_file(dict_path)
g2p_model = TransformerBlock(config=config_g2p, tokenizer=tokenizer_g2p)
g2p_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

G2P = GraphemeToPhoneme(g2p_model, tokenizer_g2p)

if __name__ == '__main__':
    print(G2P('NIKITA'))  # Expected output:['N', 'IH', 'K', 'IY', 'T', 'AH']
