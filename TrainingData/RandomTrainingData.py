import tensorflow as tf
import numpy as np


def batchEncodeData(inputIds):
    maxLen = max([len(ids) for ids in inputIds])
    paddedIds, paddedAttention = [], []
    for ids in inputIds:
        paddedIds.append(ids + [0] * (maxLen - len(ids)))
        paddedAttention.append([1] * len(ids) + [0] * (maxLen - len(ids)))

    return tf.convert_to_tensor(paddedIds, tf.int32), tf.convert_to_tensor(paddedAttention, tf.float32)


def generateRandomTrainingData(numBatches, negativeK, maxLength, maxWord, startWord=101, endWord=102):
    padd = lambda x: [startWord] + x + [endWord]
    inputIds1, inputIds2, labels = [], [], []

    for i in range(numBatches):
        sentLengths = np.random.randint(0, maxLength, negativeK + 1)
        sentLengths2 = np.random.randint(0, maxLength, negativeK)
        sents1 = [padd(np.random.randint(0, maxWord, l).tolist()) for l in sentLengths]
        sents2 = [sents1[0]] + [padd(np.random.randint(0, maxWord, l).tolist()) for l in sentLengths2]

        labels.extend([1] + [0] * negativeK)
        inputIds1.extend(sents1)
        inputIds2.extend(sents2)

    inputIds1, attention1 = batchEncodeData(inputIds1)
    inputIds2, attention2 = batchEncodeData(inputIds2)
    return inputIds1, attention1, inputIds2, attention2, tf.convert_to_tensor(labels, tf.float32)
