import re


def preprocess_text(text):
    """
    Приведение к нормальному виду с отделенными точками и запятыми
    srs:
        Hello, World! This is a sample text with numbers 12345 and symbols #$%.
    return:
        ['HELLO', ',', 'WORLD', 'THIS', 'IS', 'A', 'SAMPLE', 'TEXT', 'WITH', 'NUMBERS', 'AND', 'SYMBOLS', '.']

    """
    text = text.upper()
    text = re.sub(r'([.,])', r' \1 ', text)

    text = re.sub(r'[^A-Z .,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    text = text.split()

    return text


if __name__ == "__main__":
    sample_text = "Hello, World! This is a sample text with numbers 12345 and symbols #$%."
    processed_text = preprocess_text(sample_text)
    print("Processed text:", processed_text)  # ['HELLO', ',', 'WORLD', 'THIS', 'IS', 'A', 'SAMPLE', 'TEXT', 'WITH',
    # 'NUMBERS', 'AND', 'SYMBOLS', '.']
