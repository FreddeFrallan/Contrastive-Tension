import os

gpu = input("GPU:")
#gpu = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

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
    batch_size = 16
    negative_k = 7
    fetch_size = 500
    max_sent_len = 75

    model_name = "distilbert-base-uncased"
    tokenizer_name = "distilbert-base-uncased"

    dummy_corpus = generateDummyCorpus()
    print(dummy_corpus[:2])

    Training.tensorflowContrastiveTension(model_name, tokenizer_name, dummy_corpus, evalFunc=Evaluation.evaluateSTS,
                                          batch_size=batch_size, negative_k=negative_k, fetch_size=fetch_size,
                                          max_sent_len=max_sent_len)
