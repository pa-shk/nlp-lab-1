import math
import random
from collections import Counter, deque, namedtuple
from typing import Dict, Generator, List, Optional, Sequence, Set, Tuple

from spellchecking import EditDistanceSpellchecker

# abstraction representing possible next token
NextToken = namedtuple('NextToken', ['prob', 'token'])


class LanguageModel:
    """
    A class representing an n-gram based language model with spellchecking and text generation capabilities.

    Attributes:
        n (int): The order of n-grams used by the model.
        prev2next (Dict[Tuple[str, ...], List[NextToken]]): A mapping from prefix tuples
        to lists of possible next tokens {prefix: [possible_next_1, possible_next_2, ...]}
        ngrams_list (List[Tuple[str, ...]]): A list of all n-grams generated from the corpus.
        ngrams_counts (Counter): A Counter object counting occurrences of each n-gram in the corpus.
        vocabulary (Set[str]): A set of all unique tokens in the corpus.
        spellchecker (EditDistanceSpellchecker): A spellchecker object to handle out-of-vocabulary tokens.
    """

    def __init__(self, n: int = 3, spellchecker: Optional[EditDistanceSpellchecker] = None) -> None:
        """
        Initializes the LanguageModel with a specified n-gram order and an optional spellchecker.

        Parameters:
            n (int): The order of n-grams to be used by the model.
            spellchecker (Optional[EditDistanceSpellchecker]): An optional spellchecker object. If none is provided,
                an EditDistanceSpellchecker instance will be created.
        """
        self.n = n
        self.prev2next: Dict[Tuple[str, ...], List[NextToken]] = {}
        self.ngrams_list: List[Tuple[str, ...]] = []
        self.ngrams_counts: Counter = Counter()
        self.vocabulary: Set[str] = set()
        self.spellchecker = spellchecker if spellchecker else EditDistanceSpellchecker()

    def get_ngrams(self, tokens: Sequence[str]) -> None:
        """
        Generates n-grams of order n and lower from a given sequence of tokens and stores them in the model.

        Parameters:
            tokens (Sequence[str]): A sequence of tokens from which to generate n-grams.
        """
        for i in range(len(tokens)):
            for j in range(1, self.n + 1):
                if i + j > len(tokens):
                    continue
                self.ngrams_list.append(tuple(tokens[i: i + j]))

    def fit(self, tokens: Sequence[str]) -> None:
        """
        Trains the language model on a given sequence of tokens by generating n-grams, counting their occurrences,
        and calculating next token probabilities.

        Parameters:
            tokens (Sequence[str]): A sequence of tokens to train the model on.
        """
        self.get_ngrams(tokens)
        self.ngrams_counts = Counter(self.ngrams_list)

        for ngram, count in self.ngrams_counts.items():
            *prev, nxt = ngram
            prob = count / len(tokens)
            next_token = NextToken(token=nxt, prob=prob)
            self.prev2next.setdefault(tuple(prev), []).append(next_token)

        for v in self.prev2next.values():  # sort possible next tokens based on their probability
            v.sort()

        self.spellchecker.set_vocabulary(tokens)
        for token in tokens:
            self.vocabulary.add(token)

    def estimate_perplexity(self, tokens: Sequence[str], smoothing: Optional[str] = None) -> float:
        """
        Estimates the perplexity of a given sequence of tokens based on the model, optionally applying smoothing.

        Parameters:
            tokens (Sequence[str]): The sequence of tokens for which to estimate perplexity.
            smoothing (Optional[str]): The smoothing technique to use. Supported values are None, 'add_one',
                'linear_interpolation', and 'backoff'. If None, no smoothing is applied.

        Returns:
            float: The estimated perplexity of the sequence.

        Raises:
            NotImplementedError: If an unsupported smoothing technique is specified.
        """
        total_entropy, m = 0, 0

        # iterate through all sequence with sliding window
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i: i + self.n])
            pref = ngram[:-1]

            if not smoothing:
                if not self.ngrams_counts[pref]:  # unseen ngram leads to zero
                    return float('inf')
                estimated_prob = (self.ngrams_counts[ngram] + 1)

            elif smoothing == 'add_one':
                estimated_prob = (self.ngrams_counts[ngram] + 1) / (
                        self.ngrams_counts[pref] + len(self.vocabulary))

            elif smoothing == 'linear_interpolation':
                total_prob, c = 0, 0
                # look for all n-gram of lower order
                for j in range(len(ngram) - 1):
                    total_prob += self.ngrams_counts[ngram[j:]] / self.ngrams_counts[ngram[j: -1]]
                    c += 1
                estimated_prob = total_prob / c  # use the average
                if not total_prob:  # unseen unigram still leads to zero
                    return float('inf')

            elif smoothing == 'backoff':
                # look for all n-gram of lower order
                for j in range(len(ngram) - 1):
                    if self.ngrams_counts[ngram[j: -1]]:
                        estimated_prob = self.ngrams_counts[ngram[j:]] / self.ngrams_counts[ngram[j: -1]]
                        break  # use the probability calculated for the longest n-gram
                if not estimated_prob:
                    return float('inf')  # unseen unigram still leads to zero
            else:
                raise NotImplementedError(
                    'This smoothing technique not implemented yet, check the documentation for smoothing available')

            total_entropy -= math.log(estimated_prob, 2)
            m += 1

        return 2 ** (total_entropy / m)

    def get_random_ngram(self) -> Tuple[str, ...]:
        """
        Returns a random n-gram from the model's vocabulary.

        Returns:
            Tuple[str, ...]: A randomly selected n-gram.
        """
        rand_idx = random.randint(0, len(self.ngrams_list) - 1)
        return self.ngrams_list[rand_idx]

    def generate(self,
                 prompt: Optional[str] = None,
                 mode: str = 'max',
                 temperature: float = 0.0) -> Generator[str, None, None]:
        """
        Generates text endlessly, starting with an optional prompt, according to a specified mode and temperature.

        Parameters:
            prompt (Optional[str]): An optional starting token for text generation. If not provided, a random n-gram is used.
            mode (str): The mode of text generation. Supported modes are 'max', 'random', and 'random_weighted'.
            temperature (float): A factor controlling randomness in 'random_weighted' mode, ignored in other modes.
                When set to 1 random_weighted works as random

        Returns:
            Generator[str, None, None]: A generator yielding tokens of generated text.

        Raises:
            NotImplementedError: If an unsupported generation mode is specified.
        """
        if not prompt:
            prompt = self.get_random_ngram()
        else:
            if prompt not in self.vocabulary:
                prompt = self.spellchecker.find_closest(prompt)
            prompt = (prompt,)
        prev_tokens = deque(prompt)
        # trying to use previous context as much as possible
        while True:
            while not (possible_nxt := self.prev2next.get(tuple(prev_tokens))):
                prev_tokens.popleft()  # if previous ngram is not in counts, look for (n-1)gram
                if not prev_tokens:
                    break
            if mode == 'max':
                prob, cur_token = possible_nxt[-1]
            elif mode == 'random':
                random_idx = random.randint(0, len(possible_nxt) - 1)
                prob, cur_token = possible_nxt[random_idx]
            elif mode == 'random_weighted':
                max_prob = possible_nxt[-1][0]
                pool = [token for token in possible_nxt
                        if token.prob >= random.uniform(0, max_prob - max_prob * temperature)]
                random_idx = random.randint(0, len(pool) - 1)
                prob, cur_token = pool[random_idx]
            else:
                raise NotImplementedError(
                    'This generation mode not implemented yet, check the documentation for mode available')

            prev_tokens.append(cur_token)
            yield cur_token

    def beam_generate(self, prompt: Optional[str] = None, n_beams: int = 3, length: int = 10) -> str:
        """
        Generates text using beam search, starting with an optional prompt. Currently, supports only bigrams.

        Parameters:
            prompt (Optional[str]): An optional starting token for text generation. If not provided, a random n-gram is used.
            n_beams (int): The number of beams to maintain during generation.
            length (int): The desired length of the generated text.

        Returns:
            str: The generated text.
        """
        prompt = (prompt,) if prompt else self.get_random_ngram()
        seq_prob = {prompt: 1}
        while True:
            # add possible next tokens for all existing beams
            for seq in list(seq_prob):
                possible_next = self.prev2next.get((seq[-1],))
                for prob, new_token in possible_next:
                    seq_prob[seq + (new_token,)] = seq_prob[seq] * prob
                del seq_prob[seq]
            # prune the least probable beams
            seq_prob = {k: v for k, v
                        in sorted(seq_prob.items(), key=lambda x: -x[1])[:n_beams]}
            # return the most probable
            if len(seq) > length:
                max_prob = max(seq_prob.values())
                for k, v in seq_prob.items():
                    if v == max_prob:
                        return ' '.join(k)
