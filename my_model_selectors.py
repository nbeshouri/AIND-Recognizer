import math
import warnings
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
from typing import Dict, Tuple, List, Callable, Union


class ModelSelector(object):
    """
    Base class for model selection (strategy design pattern).

    """

    def __init__(self, all_word_sequences: Dict[str, List], all_word_Xlengths: Dict[str, Tuple], this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self) -> GaussianHMM:
        raise NotImplementedError

    def base_model(self, num_states) -> Union[GaussianHMM, None]:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

    def find_best_model(self, score_func: Callable[[int], float], selection_func: Callable) -> GaussianHMM:
        """
        Return the best `GaussianHMM` for the word.

        This is just a helper method to prevent having to duplicate the
        decision logic in each `ModelSelector` subclass.

        Args:
            score_func: A function to score models. It should take the number
                of state as an int and return the score as float or return
                `None` if a score can't be computed due to a model error.
            selection_func: A decision function to choose between models,
                either `max` or `min`.

        Returns:
            The best `GaussianHMM` for the word.

        """
        scores = ((score_func(i), i) for i in range(self.min_n_components, self.max_n_components + 1))
        scores = [score for score in scores if score[0]]
        if scores:
            # If if the score function was able to score at least one
            # model, best one.
            _, best_n = selection_func(scores, key=lambda score: score[0])
            return self.base_model(best_n)
        else:
            # If all else fails, return a 3 state model.
            return self.base_model(3)


class SelectorConstant(ModelSelector):
    """
    Select the model with value self.n_constant.

    """

    def select(self) -> GaussianHMM:
        """
        Select based on n_constant value.

        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """
    Select the model with the lowest Bayesian Information Criterion(BIC) score.

    See: http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf

    Bayesian information criteria: BIC = -2 * logL + p * logN

    """

    def select(self) -> GaussianHMM:
        """
        Select the best model for self.this_word based on BIC score for
        n between self.min_n_components and self.max_n_components.

        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection based on BIC scores

        def bic_score(num_states: int) -> Union[float, None]:
            try:
                model = self.base_model(num_states)
                log_l = model.score(self.X, self.lengths)
            except:
                return None
            num_features = model.n_features
            num_params = num_states * (num_states - 1) + (num_states - 1) + 2 * num_features * num_states
            num_data_points = len(self.X)
            return (-2. * log_l) + (num_params * math.log(num_data_points))

        return self.find_best_model(bic_score, min)


class SelectorDIC(ModelSelector):
    """
    Select best model based on Discriminative Information Criterion.

    See: Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf

    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

    """

    def select(self) -> GaussianHMM:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection based on DIC scores

        def dic_score(num_states: int) -> Union[float, None]:
            try:
                model = self.base_model(num_states)
                log_l = model.score(self.X, self.lengths)
                total_anti_log_l = sum(model.score(*X_lengths) for word, X_lengths in self.hwords.items() if word != self.this_word)
            except:
                return None
            return log_l - (total_anti_log_l / (len(self.words) - 1))

        return self.find_best_model(dic_score, max)


class SelectorCV(ModelSelector):
    """
    Select best model based on average log Likelihood of cross-validation folds.

    """

    def select(self) -> GaussianHMM:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection using CV

        def cv_score(num_states: int) -> Union[float, None]:
            if len(self.sequences) > 1:
                split_method = KFold(n_splits=min(len(self.sequences), 3))
            else:
                return None

            scores = []
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                model = GaussianHMM(n_components=num_states,
                                    covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state,
                                    verbose=False)
                try:
                    model.fit(train_X, train_lengths)
                    scores.append(model.score(test_X, test_lengths))
                except:
                    return None
            return np.mean(scores)

        return self.find_best_model(cv_score, max)
