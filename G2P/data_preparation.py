import re


def intToWord(number):
    ones = ("", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    tens = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
    teens = (
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
    levels = (
        "", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion", "sextillion", "septillion",
        "octillion", "nonillion")

    word = ""
    num = reversed(str(number))
    number = ""
    for x in num:
        number += x
    del num
    if len(number) % 3 == 1: number += "0"
    x = 0
    for digit in number:
        if x % 3 == 0:
            word = levels[x // 3] + " " + word
            n = int(digit)
        elif x % 3 == 1:
            if digit == "1":
                num = teens[n]
            else:
                num = tens[int(digit)]
                if n:
                    if num:
                        num +=  ones[n]
                    else:
                        num = ones[n]
            word = num + " " + word
        elif x % 3 == 2:
            if digit != "0":
                word = ones[int(digit)] + " hundred " + word
        x += 1
    return word.strip(" ")


def preprocess_text(text):
    """
    Приведение к нормальному виду с отделенными точками и запятыми
    srs:
        Hello, World! This is a sample text with numbers 12345 and symbols #$%.
    return:
        ['HELLO', ',', 'WORLD', 'THIS', 'IS', 'A', 'SAMPLE', 'TEXT', 'WITH', 'NUMBERS', 'AND', 'SYMBOLS', '.']
    """
    if not(text.isspace()) and text and text:

        text = text.upper()
        text = re.sub(r'([.,])', r' \1 ', text)

        text = re.sub(r'[^A-Z .,^0-9]', '', text)
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
    sample_text = "Hello, World! This is a sample text with numbers 12345 and symbols #$%."
    processed_text = preprocess_text(sample_text)
    print("Processed text:", processed_text)
