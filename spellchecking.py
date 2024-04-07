from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Set, Tuple

import fasttext


class SpellChecker(ABC):
    """
    Abstract base class for a spellchecker.
    """

    def __init__(self) -> None:
        self._vocabulary: Optional[Set[str]] = None

    def set_vocabulary(self, vocabulary: Set[str]) -> None:
        """
        Set the vocabulary used by the spellchecker.

        Args:
            vocabulary (Set[str]): The set of words in the vocabulary.
        """
        self._vocabulary = set(vocabulary)

    @abstractmethod
    def find_closest(self, word: str) -> str:
        """
        Find the closest word in the vocabulary to the given word.

        Args:
            word (str): The word to find the closest match for.

        Returns:
            str: The closest word in the vocabulary.
        """
        pass


class EditDistanceSpellchecker(SpellChecker):
    """
    Spell checker implementation based on edit distance.
    """

    @staticmethod
    def _calculate_edit_distance(word1: str, word2: str) -> int:
        """
        Calculate the edit distance between two words using dynamic programming.

        Args:
            word1 (str): The first word.
            word2 (str): The second word.

        Returns:
            int: The edit distance between the two words.
        """
        word1 = ' ' + word1
        word2 = ' ' + word2
        dp = [[0 for _ in range(len(word1))] for _ in range(len(word2))]
        for col in range(len(word1)):
            dp[0][col] = col
        for row in range(len(word2)):
            dp[row][0] = row
        for row in range(1, len(word2)):
            for col in range(1, len(word1)):
                if word1[col] == word2[row]:
                    dp[row][col] = dp[row - 1][col - 1]
                else:
                    dp[row][col] = min(
                        dp[row - 1][col],
                        dp[row][col - 1],
                        dp[row - 1][col - 1]
                        ) + 1
        return dp[-1][-1]

    def find_closest(self, word: str) -> str:
        """
        Find the closest word in the vocabulary to the given word based on edit distance.

        Args:
            word (str): The word to find the closest match for.

        Returns:
            str: The closest word in the vocabulary.
        """
        best_match, min_dist = '', float('inf')
        for another_word in self._vocabulary:
            dist = self._calculate_edit_distance(word, another_word)
            if dist < min_dist:
                best_match = another_word
                min_dist = dist
        return best_match


class FastTextSpellChecker(SpellChecker):
    """
    Spell checker implementation using FastText word embeddings.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model: Optional[fasttext.FastText._FastText] = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the FastText model from the 'cc.ru.300.bin' file.
        """
        path = str(Path().absolute() / 'cc.ru.300.bin')
        self._model = fasttext.load_model(path)

    def find_closest(self, word: str) -> str:
        """
        Find the closest word in the vocabulary to the given word based on FastText word embeddings.

        Args:
            word (str): The word to find the closest match for.

        Returns:
            str: The closest word in the vocabulary.
        """
        neighbors: List[Tuple[float, str]] = self._model.get_nearest_neighbors(word)
        for dist, neighbor in neighbors:
            if neighbor in self._vocabulary:
                return neighbor