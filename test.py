from G2P_lexicon import g2p_en_lexicon
text = "text, numbers, and some strange symbols !№;% 21"
g2p = g2p_en_lexicon()
phonemes = g2p(text, with_stress=False)
print(phonemes)