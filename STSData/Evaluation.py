from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
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


def evaluateOnData(contrastiveModel, textEncodeFunc, dataset, batchSize=512):
    texts = Dataset._loadUniqueCaptions(dataset, True)
    inputIds, attention = textEncodeFunc(texts)
    sent2VecModel1, sent2VecModel2 = {}, {}

    def addInSent2Vec(texts, embeddings, targetDict):
        for txt, emb in zip(texts, embeddings):
            targetDict[txt] = emb

    f = lambda x, i: x[i:i + batchSize]
    for i in tqdm.tqdm(range(0, len(texts), batchSize), "Generating Eval Embeddings"):
        batchTexts, batchInds, batchAtt = f(texts, i), f(inputIds, i), f(attention, i)
        batchEmbs1, batchEmbs2 = contrastiveModel((batchInds, batchAtt), training=False)
        addInSent2Vec(batchTexts, batchEmbs1, sent2VecModel1)
        addInSent2Vec(batchTexts, batchEmbs2, sent2VecModel2)

    return evalCorrelationScores(sent2VecModel1, dataset), evalCorrelationScores(sent2VecModel2, dataset)
