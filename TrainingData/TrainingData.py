from STSData import Dataset as STSDataset
from TrainingData import Utils
import tensorflow as tf
import numpy as np
import pickle

MAIN_CORPUS_PATH = "TrainingCorpus/"


def loadCustomWikiDumpTexts(dumpName, flattenIntoSentences=True):
    with open(dumpName, 'rb') as fp:
        if (flattenIntoSentences == False):
            return pickle.load(fp)
        allSentences = []
        for section in pickle.load(fp):
            allSentences.extend(section)
        return allSentences


def _loadArabicWikiTexts():
    return loadCustomWikiDumpTexts(MAIN_CORPUS_PATH + 'ArabicWikipediaTexts-Clean.pkl')


def _loadEnglishWikiTexts():
    return loadCustomWikiDumpTexts(MAIN_CORPUS_PATH + 'EnglishWikipediaTexts-Clean.pkl')


def _loadRussainWikiTexts():
    return loadCustomWikiDumpTexts(MAIN_CORPUS_PATH + 'RussainWikipediaTexts-Clean.pkl')


def _loadSpanishWikiTexts():
    return loadCustomWikiDumpTexts(MAIN_CORPUS_PATH + 'SpanishWikipediaTexts-Clean.pkl')


def _loadSwedishWikiTexts():
    return loadCustomWikiDumpTexts(MAIN_CORPUS_PATH + "SwedishhWikipediaTexts-Clean.pkl")


def _prepareSelectedSamples(positiveSentences, negSentences, negativeK):
    inTexts1, inTexts2, labels, negCounter = [], [], [], 0
    for s in positiveSentences:
        inTexts2.append(s)
        inTexts2.extend(negSentences[negCounter:negCounter + negativeK])

        inTexts1.append(s)
        inTexts1.extend(negSentences[negCounter + negativeK:negCounter + negativeK * 2])
        negCounter += negativeK * 2

        labels.extend([1] + [0] * negativeK)

    inds1, att1 = Utils.batchEncodeData(inTexts1)
    inds2, att2 = Utils.batchEncodeData(inTexts2)

    f = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)
    return inds1, att1, inds2, att2, f(labels)


# Ugly arguments hack
def generateRandomTrainingExamples(tokenizer, data, numBatches, negativeK, maxLength, maxWord, startWord=101,
                                   endWord=102):
    padd = lambda x: [startWord] + x.tolist() + [endWord]

    positiveSentLengths = np.random.randint(0, maxLength, numBatches)
    negativeSentLengths = np.random.randint(0, maxLength, numBatches * negativeK * 2)

    posSentences = [padd(np.random.randint(0, maxWord, sentLen)) for sentLen in positiveSentLengths]
    negSentences = [padd(np.random.randint(0, maxWord, sentLen)) for sentLen in negativeSentLengths]
    return _prepareSelectedSamples(posSentences, negSentences, negativeK)


# Ugly arguments hack
def generateTrainingSamples(tokenizer, data, numBatches, negativeK=7, maxLength=200, **kwargs):
    positiveSentences = [data[i] for i in np.random.choice(range(len(data)), numBatches)]
    negSentences = [data[i] for i in np.random.choice(range(len(data)), numBatches * negativeK * 2)]

    def fixLength(sentences):
        for i, sent in enumerate(sentences):
            while len(sent) > maxLength:
                sent = data[np.random.randint(0, len(data))]
                sentences[i] = sent

    fixLength(positiveSentences)
    fixLength(negSentences)

    # Convert Text into input ids. #TODO There exists far better ways of organizing and doing this
    positiveSentences = \
        tokenizer.batch_encode_plus(positiveSentences, add_special_tokens=True, max_length=maxLength, truncation=True)[
            'input_ids']
    negSentences = \
        tokenizer.batch_encode_plus(negSentences, add_special_tokens=True, max_length=maxLength, truncation=True)[
            'input_ids']

    return _prepareSelectedSamples(positiveSentences, negSentences, negativeK)


Corpuses = {
    'arabic': (
        _loadArabicWikiTexts, generateTrainingSamples, STSDataset.loadDevData),

    'english': (
        _loadEnglishWikiTexts, generateTrainingSamples, STSDataset.loadDevData),

    'russain': (_loadRussainWikiTexts, generateTrainingSamples, STSDataset.loadDevData),

    'random': (
        lambda: [], generateRandomTrainingExamples, STSDataset.loadDevData),

    'spanish': (
        _loadSpanishWikiTexts, generateTrainingSamples, STSDataset.loadDevData),

    'swedish': (
        _loadSwedishWikiTexts, generateTrainingSamples, STSDataset.loadDevData,
    )
}
