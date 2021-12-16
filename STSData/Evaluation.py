'''
This is just a simple script to get you up and running.
If you are aiming to publish your own results, please consider relying on SentEval for evaluation.
https://github.com/facebookresearch/SentEval
'''

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
from ContrastiveTension import Inference
from STSData import Dataset
import numpy as np
import tqdm


def evalCorrelationScores(sent2Vecs, dataset):
    similarityScores, humanScores = [], []
    for i, data in enumerate(dataset):
        s1, s2, score = data
        humanScores.append(score)
        similarityScores.append(cosine_similarity([sent2Vecs[s1]], [sent2Vecs[s2]])[0][0])

    x, y = np.array(similarityScores), np.array(humanScores)
    pearResults = pearsonr(x, y)
    spearResults = spearmanr(x, y)

    return {'Pearson': pearResults[0], 'Spearman': spearResults[0]}


def evaluateSTS(model, tokenizer, batch_size=512, use_dev_data=False):
    data = Dataset.loadDevData() if use_dev_data else Dataset.loadTestData()
    texts = Dataset.getUniqueCaptions(data)

    sent2Vec = {}
    for i in tqdm.tqdm(range(0, len(texts), batch_size), "Generating Eval Embeddings"):
        batchTexts = texts[i:i + batch_size]
        embs = Inference.tensorflowGenerateSentenceEmbeddings(model, tokenizer, batchTexts)

        for txt, emb in zip(batchTexts, embs):
            sent2Vec[txt] = emb

    return evalCorrelationScores(sent2Vec, data)
