# Grapheme to Phoneme (G2P) with Stress

This project provides a Grapheme to Phoneme (G2P) conversion tool that first checks the CMU Pronouncing Dictionary for phoneme translations. If a word is not found in the dictionary, it utilizes two Transformer-based models to generate phoneme translations and add stress markers. The output is in ARPAbet format, and the model can also convert graphemes into phoneme integer indices.

## Features

1. **CMU Pronouncing Dictionary Integration**: First checks the CMU dictionary for phoneme translations.
2. **Transformer-Based Conversion**:
    - **Phoneme Generation**: The first Transformer model converts graphemes into phonemes.
    - **Stress Addition**: The second Transformer model adds stress markers to the phonemes.
3. **ARPAbet Output**: Outputs phonemes in ARPAbet format.
4. **Phoneme Integer Indices**: Converts graphemes to phoneme integer indices.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/NikiPshg/G2P_en_lex.git
    cd G2P_en_lex
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requiremenst.txt
    ```


### Example

```python
from G2P import G2P_en_lex

# Initialize the G2P converter
g2p_converter = G2P_en_lex()

# Convert a word to phonemes
text = "text, numbers, and some strange symbols !№;% 21"
phonemes = G2P_en_lex(text, with_stress=False)
['T', 'EH', 'K', 'S', 'T', ' ', ',', ' ', 'N', 'AH', 'M', 'B', 'ER', 'Z', ' ', ',', ' ',
'AH', 'N', 'D', ' ', 'S', 'AH', 'M', ' ', 'S', 'T', 'R', 'EY', 'N', 'JH', ' ',
'S', 'IH', 'M', 'B', 'AH', 'L', 'Z', ' ',
'T', 'W', 'EH', 'N', 'IY', 'W', 'AH', 'N']
print("Phonemes:", phonemes)


