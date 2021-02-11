from ContrastiveTension import Training
from STSData import Evaluation
import numpy as np
import tqdm


def generateDummyCorpus(num_sentences=10000, max_words_per_sent=15, max_word_len=8):
    import string

    sents = []
    letters = [c for c in string.ascii_lowercase]
    words_in_sents = np.random.randint(1, max_words_per_sent, num_sentences)
    for num_words in tqdm.tqdm(words_in_sents, desc='Generating Dummy Dataset'):
        word_lens = np.random.randint(1, max_word_len, num_words)
        sents.append(' '.join([''.join(np.random.choice(letters, n)) for n in word_lens]))
    return sents

if __name__ == '__main__':
    # Hyperparamters from the Original CT paper
    batch_size = 16
    negative_k = 7
    fetch_size = 500
    max_sent_len = 75

    model_name = "distilbert-base-uncased"
    tokenizer_name = "distilbert-base-uncased"
    # This generates a list of randomly generated sentences, change this into whatever text corpus you wish to use.
    corpus = generateDummyCorpus()

    eval_func = lambda m, t, b: Evaluation.evaluateSTS(m, t, b, use_dev_data=True)
    Training.tensorflowContrastiveTension(model_name, tokenizer_name, corpus, evalFunc=eval_func,
                                          batch_size=batch_size, negative_k=negative_k, fetch_size=fetch_size,
                                          max_sent_len=max_sent_len)
