from typing import List

from nltk import regexp_tokenize, sent_tokenize


class CustomTokenizer:
    """
    A custom tokenizer implemented using NLTK.

    This tokenizer provides methods for tokenizing sentences and tokenizing a whole corpus of text.
    """

    def __init__(self, pattern: str = r'\w+\S\w+|\w', do_lower_case: bool = True,
                 add_eos_token: bool = True, add_bos_token: bool = False) -> None:
        """
        Initialize the CustomTokenizer.

        Args:
            pattern (str, optional): A regular expression pattern used for tokenization.
                Defaults to r'\w+\S\w+|\w'.
            do_lower_case (bool, optional): Whether to convert the text to lowercase before tokenization.
                Defaults to True.
            add_eos_token (bool, optional): Whether to add an end-of-sentence token ("<EOS>") at the end of each sentence.
                Defaults to True.
            add_bos_token (bool, optional): Whether to add a beginning-of-sentence token ("<BOS>") at the beginning of each sentence.
                Defaults to False.
        """
        self.do_lower_case = do_lower_case
        self.pattern = pattern
        self.add_eos_token = add_eos_token
        self.add_bos_token = add_bos_token

    def tokenize_sentence(self, text: str) -> List[str]:
        """
        Tokenize a single sentence.

        Args:
            text (str): The input sentence to tokenize.

        Returns:
            List[str]: The list of tokens extracted from the sentence.
        """
        if not self.do_lower_case:
            return regexp_tokenize(text, pattern=self.pattern)
        return regexp_tokenize(text.lower(), pattern=self.pattern)

    def tokenize_corpus(self, text: str) -> List[str]:
        """
        Tokenize a corpus of text.

        Args:
            text (str): The input text corpus to tokenize.

        Returns:
            List[str]: The list of tokens extracted from the corpus.
        """
        sentences = sent_tokenize(text)
        res = []
        for s in sentences:
            if self.add_bos_token:
                res.append('<BOS>')
            tokens = self.tokenize_sentence(s)
            res.extend(tokens)
            if not self.add_eos_token:
                continue
            res.append('<EOS>')
        return res