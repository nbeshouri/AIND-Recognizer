import warnings
from asl_data import SinglesData
import math
import io
import os
import re
import pandas as pd
from collections import deque
from functools import lru_cache
from hmmlearn.hmm import GaussianHMM
from typing import Dict, Tuple, List


lm_dataframe = None


def recognize(models: Dict[str, GaussianHMM], test_set: SinglesData, lm_weight=0, beam_size=1) -> Tuple[List, List]:
    """
    Recognize test word sequences from word models set

    Args:
        models: `dict` of trained models.
        test_set: `SinglesData` representing the test set.
        lm_weight: The multiplier applied to to the language model's log
            likelihood estimate for a word before it's added to the HMM's
            estimate. The default value of 0 effectively disables the
            language model.
        beam_size: The number of alternative guesses retained for later
            consideration as the recognizer guesses each word. This only
            affects the outcome if the language model is used. The default
            value of 1 is equivalent to a greedy, best-first search.

    Returns:
        (list, list)  as probabilities, guesses

    """
    # TODO implement the recognizer
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    guesses = []
    probabilities = []

    def get_word_log_l(model: GaussianHMM, seq_index: int, word: str, prev_words: Tuple[str]) -> float:
        """Return word log likelihood given data, HMM, and prev words."""
        lm_log_l = get_lm_log_l(word, tuple(prev_words[-2:])) if lm_weight else 0.0
        hmm_log_l = get_model_log_l(model, test_set, seq_index)
        return (lm_weight * lm_log_l) + hmm_log_l

    # Iterate over the sentences in the test set, which are represented
    # as list of sequence indexes. These indexes can be used to get
    # (X, lengths) from test_set.
    for sentence in test_set.sentences_index.values():

        def get_successors(path: Tuple[str]) -> List[Tuple[str]]:
            """Return beam_size most likely successor paths."""
            # Calulate word probs for the next word in the sentence.
            seq_index = sentence[len(path)]
            word_probs = {word: get_word_log_l(model, seq_index, word, path)
                          for word, model in models.items()}
            # Convert them into a sorted sequence of (word, word_prob) pairs.
            word_prob_pairs = sorted(word_probs.items(), key=lambda x: x[1])
            # Return successor paths for the beam_size most likely next words.
            return [path + (pair[0],) for pair in word_prob_pairs[-beam_size:]]

        # Conduct a search through space of possible sentences. If beam_size
        # is 1, this is equivalent to greedily picking the highest scoring
        # word.
        frontier = deque(get_successors(tuple()))
        candidates = []
        while frontier:
            path = frontier.popleft()
            if len(path) == len(sentence):
                candidates.append(path)
            else:
                for successor in get_successors(path):
                    frontier.append(successor)

        def get_candidate_log_l(candidate: Tuple[str]) -> float:
            """Return log likelihood of a candidate sentence."""
            return sum([get_word_log_l(models[word], sentence[i], word, candidate[:i])
                        for i, word in enumerate(candidate)])

        # Pick the best candidate sentence.
        sentence_guess = max(candidates, key=lambda c: get_candidate_log_l(c))
        # Extend the cumulative guesses and probabilities list.
        guesses.extend(sentence_guess)
        probabilities.extend([{word: get_word_log_l(model, seq_index, word, sentence_guess[:i])
                               for word, model in models.items()}
                               for i, seq_index in enumerate(sentence)])

    return probabilities, guesses


@lru_cache(maxsize=None)
def get_model_log_l(model: GaussianHMM, test_set: SinglesData, sequence_index: int) -> float:
    """Return log(P(X|model) according to the HMM."""
    try:
        X, lengths = test_set.get_item_Xlengths(sequence_index)
        return model.score(X, lengths)  # Log(P(X|model)
    except:
        return -math.inf  # Log(0)


@lru_cache(maxsize=None)
def get_lm_log_l(word: str, prev_words: Tuple[str]) -> float:
    """Return log(P(word|prev_words)) according to the language model."""

    # Load the language model Dataframe if it isn't loaded already.
    global lm_dataframe
    if lm_dataframe is None:
        lm_dataframe = get_lm_dataframe()

    # If there are no prev words or only one, append '<s>' tag used in the file
    # to indicate the start of a sentence.
    if not prev_words:
        prev_words = ('', '<s>')
    elif len(prev_words) < 2:
        prev_words = ('<s>',) + prev_words
    else:
        prev_words = prev_words[-2:]

    # Convert the word str and prev_word tuple into trigram, bigram, and
    # unigram strings.
    n_grams = [' '.join(prev_words + (word,)), ' '.join(prev_words[1:] + (word,)), word]

    # Some signs have multiple versions (e.g. 'GO1', 'GO1'). Use regex
    # to strip off the trailing digits from words in the n-grams.
    n_grams = [re.sub(r'(?<=\w)\d(?=\s|$)', '', n_gram) for n_gram in n_grams]

    # Try to find the n-grams, starting with the trigram, in the language
    # model.  Simpler n-grams will have a non-zero back-off weight added to
    # them to so that can be used in the trigram model.
    for n_gram in n_grams:
        if n_gram in lm_dataframe.index:
            likelihood = lm_dataframe.loc[n_gram]['log_likelihood']
            backoff_weight = lm_dataframe.loc[n_gram]['backoff_weight']
            # The ARPA format specifies base 10 log likelihoods, but
            # hmmlearn appears to return base e log likelihoods, so
            # change the base before returning.
            return (likelihood + backoff_weight) / math.log(math.e, 10)

    # All words should be in the lm file, at least as unigrams, so
    # if we can't find a match something has gone wrong.
    raise Exception(f'"{word}" was not found in the language model.')


def get_lm_dataframe() -> pd.DataFrame:
    """Return a `DataFrame` that represents the language model."""
    # Look for the language model file.
    path = os.path.join(os.getcwd(), 'data', 'ukn.3.lm')
    if not os.path.isfile(path):
        path = os.path.join(os.getcwd(), 'ukn.3.lm')
    if not os.path.isfile(path):
        raise FileNotFoundError('The "ukn.3.lm" language model file was '
                                'not found in the data directory.')
    with open(path, 'r') as f:
        lm_str = f.read()
    # Strip out everything but the lines containing n-gram data.
    lines = re.findall(r'^-\d.*\n', lm_str, re.MULTILINE)
    lm_str = ''.join(lines)
    # Add column names to the top of the data lines.
    lm_str = 'log_likelihood\tn_gram\tbackoff_weight\n' + lm_str
    # Parse the str into a DataFrame.
    lm_dataframe = pd.read_csv(io.StringIO(lm_str), sep='\t')
    # Replace the index with the n-gram strings.
    lm_dataframe.set_index('n_gram', inplace=True)
    lm_dataframe.sort_index()
    # Replace NaN back-off weights with zeroes.
    lm_dataframe.fillna(0.0, inplace=True)
    return lm_dataframe
