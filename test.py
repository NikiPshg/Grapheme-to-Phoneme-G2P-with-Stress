from G2P_lexicon import g2p_en_lexicon

# Initialize the G2P converter
g2p = g2p_en_lexicon()
# Convert a word to phonemes
text = "text, numbers, and some strange symbols !№;% 21"
phonemes = g2p(text, with_stress=False)




