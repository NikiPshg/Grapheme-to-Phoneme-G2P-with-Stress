import re

one = ["", "one ", "two ", "three ", "four ",
       "five ", "six ", "seven ", "eight ",
       "nine ", "ten ", "eleven ", "twelve ",
       "thirteen ", "fourteen ", "fifteen ",
       "sixteen ", "seventeen ", "eighteen ",
       "nineteen "]

# strings at index 0 and 1 are not used,
# they are to make array indexing simple
ten = ["", "", "twenty ", "thirty ", "forty ",
       "fifty ", "sixty ", "seventy ", "eighty ",
       "ninety "]


def numToWords(n, s):
    str = ""

    if n <= 19:
        str += one[n]
    # if n is more than 19, divide it
    else:
        str += ten[n // 10] + one[n % 10]

    # if n is non-zero
    if (n):
        str += s

    return str


def intToWord(n):
    """
   turning a number into a word
    srs:
        "12345"
    return:
       ['twelve thousand three hundred and forty five']
    """
    n=int(n)
    out = ""

    out += numToWords((n // 10000000),
                      "crore ")

    out += numToWords(((n // 100000) % 100),
                      "lakh ")

    out += numToWords(((n // 1000) % 100),
                      "thousand ")

    out += numToWords(((n // 100) % 10),
                      "hundred ")

    if n > 100 and n % 100:
        out += "and "

    # handles digits at ones and tens
    # places (if any)
    out += numToWords((n % 100), "")

    return out.strip()


def preprocess_text(text):
    """
    Reduction to normal form with punctuation marks
    srs:
        "Hello, World! This is a sample text with numbers 12345 and symbols :#$%."
    return:
       ['HELLO', ',', 'WORLD', '!', 'THIS', 'IS', 'A', 'SAMPLE', 'TEXT', 'WITH', 'NUMBERS', 'TWELVE', 'THOUSAND', 'THREE', 'HUNDRED', 'AND', 'FORTY', 'FIVE', 'AND', 'SYMBOLS', ':', '.']
    """
    if not (text.isspace()) and text and text:

        text = text.upper()
        text = re.sub(r'([.,:;?!])', r' \1 ', text)

        text = re.sub(r'[^A-Z .,:;?!^0-9]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        text = text.split()
        result = []
        for word in text:
            if word.isdigit():
                result = result + (intToWord(word).upper()).split()
            else:
                result.append(word)
    else:
        result = ['текст введи :(']

    return result


if __name__ == "__main__":
    print(intToWord(21))
    sample_text = "Hello, World! This is a sample text with numbers 12345 and symbols :#$%."
    processed_text = preprocess_text(sample_text)
    print("Processed text:", processed_text)
