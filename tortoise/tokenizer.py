import re
from os import path

from inflect import engine
from tokenizers import Tokenizer as _Tokenizer
from torch import Tensor
from unidecode import unidecode

inflect = engine()


class CleaningPipeline:
    whitespace_re = re.compile(r"\s+")

    comma_number_re = re.compile(r"([0-9][0-9,]+[0-9])")
    decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
    pounds_re = re.compile(r"£([0-9,]*[0-9]+)")
    dollars_re = re.compile(r"\$([0-9.,]*[0-9]+)")
    ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
    number_re = re.compile(r"[0-9]+")
    # List of (regular expression, replacement) pairs for abbreviations:
    abbreviations = [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("mrs", "misess"),
            ("mr", "mister"),
            ("dr", "doctor"),
            ("st", "saint"),
            ("co", "company"),
            ("jr", "junior"),
            ("maj", "major"),
            ("gen", "general"),
            ("drs", "doctors"),
            ("rev", "reverend"),
            ("lt", "lieutenant"),
            ("hon", "honorable"),
            ("sgt", "sergeant"),
            ("capt", "captain"),
            ("esq", "esquire"),
            ("ltd", "limited"),
            ("col", "colonel"),
            ("ft", "fort"),
        ]
    ]

    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text

    def ascii(self):
        self.text = unidecode(self.text)
        return self

    def lower(self):
        self.text = self.text.lower()
        return self

    def expand_numbers(self):
        def remove_commas(match: re.Match) -> str:
            return match.group(1).replace(",", "")

        def expand_decimal_point(match: re.Match) -> str:
            return match.group(1).replace(".", " point ")

        def expand_dollars(match: re.Match) -> str:
            parts = match.group(1).split(".")
            if len(parts) > 2:
                return match.group(1) + " dollars"
            dollars = int(parts[0] or 0)
            cents = int(parts[1]) if len(parts) == 2 and parts[1] else 0
            if dollars and cents:
                dollar_unit = "dollar" if dollars == 1 else "dollars"
                cent_unit = "cent" if cents == 1 else "cents"
                return f"{dollars} {dollar_unit}, {cents} {cent_unit}"
            if dollars:
                dollar_unit = "dollar" if dollars == 1 else "dollars"
                return f"{dollars} {dollar_unit}"
            if cents:
                cent_unit = "cent" if cents == 1 else "cents"
                return f"{cents} {cent_unit}"
            return "zero dollars"

        def expand_ordinal(match: re.Match) -> str:
            return inflect.number_to_words(match.group(1))

        # NOTE: I'm scratching the custom handling here, inflect is probably just good enough
        def expand_number(match: re.Match) -> str:
            num = int(match.group(0))
            return inflect.number_to_words(num, andword="")

        text = re.sub(self.comma_number_re, remove_commas, self.text)
        text = re.sub(self.pounds_re, r"\1 pounds", text)
        text = re.sub(self.dollars_re, expand_dollars, text)
        text = re.sub(self.decimal_number_re, expand_decimal_point, text)
        text = re.sub(self.ordinal_re, expand_ordinal, text)
        self.text = re.sub(self.number_re, expand_number, text)
        return self

    def expand_abbreviations(self):
        for regex, replacement in self.abbreviations:
            self.text = re.sub(regex, replacement, self.text)
        return self

    def collapse_whitespace(self):
        self.text = re.sub(self.whitespace_re, " ", self.text)
        return self


class Tokenizer:
    def __init__(self):
        self.tokenizer = _Tokenizer.from_file(path.join(path.dirname(__file__), "tokenizer.json"))

    def encode(self, text: str):
        pipeline = CleaningPipeline(text).ascii().lower().expand_numbers().expand_abbreviations().collapse_whitespace()
        processed_text = str(pipeline).replace('"', "").replace(" ", "[SPACE]")
        return self.tokenizer.encode(processed_text).ids

    # TODO: jaxtype
    def decode(self, sequence: Tensor):
        text: str = self.tokenizer.decode(sequence.cpu().numpy(), skip_special_tokens=False)
        text = text.replace(" ", "").replace("[SPACE]", " ").replace("[STOP]", "").replace("[UNK]", "")
        return text


if __name__ == "__main__":
    import torch

    tokenizer = Tokenizer()
    encoded_ids = tokenizer.encode("Mr. Smith would like 9,999 pies for $88.54, please")
    decoded = tokenizer.decode(torch.tensor(encoded_ids))
    print(decoded)
