import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for item_index in range(test_set.num_items):
        item_X, item_lengths = test_set.get_item_Xlengths(item_index)
        item_probabilities = {}
        for word in models:
            model = models[word]
            try:
                logLvalue = model.score(item_X, item_lengths)
                item_probabilities[word] = logLvalue
            except:
                # print("Error scoring model for word: {} and item_index: {}".format(word, item_index))
                item_probabilities[word] = float('-inf')
                pass
        best_guess = max(item_probabilities, key=item_probabilities.get)
        probabilities.append(item_probabilities)
        guesses.append(best_guess)


    return probabilities, guesses
