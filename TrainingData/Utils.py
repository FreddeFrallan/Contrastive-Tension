import tensorflow as tf


def batchEncodeData(inputIds):
    maxLen = max([len(ids) for ids in inputIds])
    paddedIds, paddedAttention = [], []
    for ids in inputIds:
        paddedIds.append(ids + [0] * (maxLen - len(ids)))
        paddedAttention.append([1] * len(ids) + [0] * (maxLen - len(ids)))

    return tf.convert_to_tensor(paddedIds, tf.int32), tf.convert_to_tensor(paddedAttention, tf.float32)
