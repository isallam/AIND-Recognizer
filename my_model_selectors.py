import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
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

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
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


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN

        p is calculated as:
           p =  n * n + 2 * n * d - 1
           p = num_states * num_states + 2 * num_states * len(self.X[0]) - 1
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        results = []
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(num_states)
                logL = model.score(self.X, self.lengths)
                logN = np.log(len(self.X))
                params = num_states * num_states + 2 * num_states * len(self.X[0]) - 1
                BIC_score = -2 * logL + params * logN
                # print("word: {}, num_states: {}, BIC: {}".format(self.this_word, num_states, BIC_score))
                results.append((BIC_score, model))
            except:
                # print("Error training model for word: {} with num_states: {}".format(word, num_states))
                pass

        if results != []:
            score, model = min(results)
            return model
        else:
            return None

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

    From some comment:
        DIC = log(P(original word)) - average(log(P(otherwords)))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        results = []
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(num_states)
                logL_word = model.score(self.X, self.lengths)
                logL_others = []
                for word in self.words:
                    if word == self.this_word:
                        continue
                    other_word_X, other_word_lengths = self.hwords[word]
                    logL_other_score = model.score(other_word_X, other_word_lengths)
                    logL_others.append(logL_other_score)
                avg_logL_others = np.average(logL_others)
                DIC_score = logL_word - avg_logL_others
                # print("word: {}, num_states: {}, DIC: {}".format(self.this_word, num_states, DIC_score))
                results.append((DIC_score, model))
            except:
                # print("Error training model for word: {} with num_states: {}".format(word, num_states))
                pass

        if results != []:
            score, model = max(results)
            return model
        else:
            return None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        split_method = KFold()
        word = self.this_word
        # print("Word:{}".format(word))
        results = []
        try:
            cv_train_sets = []
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                cv_train_sets.append(cv_test_idx)
            for num_states in range(self.min_n_components, self.max_n_components + 1):
                try:
                    scores = []
                    for train_idx in cv_train_sets:
                        # print("Train fold indices:{}".format(train_idx))  # view indices of the folds
                        self.X, self.lengths = combine_sequences(train_idx, self.sequences)
                        model = self.base_model(num_states)
                        logL = model.score(self.X, self.lengths)
                        scores.append(logL)
                    avgLogLikelihood = np.average(scores)
                    # print("Word:{}, num_states:{} => scores:{}, avg:{}".format(word, num_states, scores, avgLogLikelihood))
                    results.append((avgLogLikelihood, model))
                except:
                    # print("Error training model for word: {} with num_states: {}".format(word, num_states))
                    pass
        except ValueError as valueError:
            # print("Error spliting word: {} - error: {}".format(word, valueError))
            pass

        if results != []:
            score, model = max(results)
            return model
        else:
            return None