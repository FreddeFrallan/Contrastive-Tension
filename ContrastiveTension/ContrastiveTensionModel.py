'''
Please note that this code is optimized towards comprehension and not performance.
'''

from tensorflow.python.keras import backend as K
import tensorflow as tf
import numpy as np
import tqdm


class ContrastivTensionModel(tf.keras.Model):

    def __init__(self, model1, model2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model1 = model1
        self.model2 = model2
        self.loss = tf.losses.BinaryCrossentropy(from_logits=True)
        self.nonReductionLoss = lambda y, x: K.binary_crossentropy(y, x, from_logits=True)

    def generateSingleEmbedding(self, model, inData, training=False):
        inds, att = inData
        embs = model({'input_ids': inds, 'attention_mask': att}, training=training)[0]
        outAtt = tf.cast(att, tf.float32)
        sampleLength = tf.reduce_sum(outAtt, axis=-1, keepdims=True)
        maskedEmbs = embs * tf.expand_dims(outAtt, axis=-1)
        return tf.reduce_sum(maskedEmbs, axis=1) / tf.cast(sampleLength, tf.float32)

    @tf.function
    def call(self, inputs, training=False, mask=None):
        emb1 = self.generateSingleEmbedding(self.model1, inputs, training)
        emb2 = self.generateSingleEmbedding(self.model2, inputs, training)
        return emb1, emb2

    @tf.function
    def predictandCompareSents(self, x1, x2, training=False):
        emb1 = self.generateSingleEmbedding(self.model1, x1, training)
        emb2 = self.generateSingleEmbedding(self.model2, x2, training)
        return self.compareSents(emb1, emb2), emb1, emb2

    def compareSents(self, emb1, emb2):
        return tf.reduce_sum(emb1 * emb2, axis=-1)

    def extractPositiveAndNegativeLoss(self, predValues, labels):
        losses = self.nonReductionLoss(labels, predValues)
        pLoss = tf.reduce_sum(losses * labels)
        nLoss = tf.reduce_sum(losses * (labels - 1) * -1)
        return pLoss, nLoss

    @tf.function
    def predictAndUpdate(self, inds1, att1, inds2, att2, labels):
        with tf.GradientTape() as tape:
            predValues, emb1, emb2 = self.predictandCompareSents((inds1, att1), (inds2, att2),
                                                                 training=False)

            cosineLoss = self.loss(labels, predValues)
            grad = tape.gradient(cosineLoss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

            # Extract loss for Positive/Negative examples for later examination
            pLoss, nLoss = self.extractPositiveAndNegativeLoss(predValues, labels)

            return cosineLoss, pLoss, nLoss

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1,
            use_multiprocessing=False, **kwargs):
        contrastiveLosses, pLosses, nLosses = [], [], []
        f = lambda x, i: x[i:i + batch_size]
        inds1, att1, inds2, att2 = x

        for i in tqdm.tqdm(range(0, len(inds1), batch_size)):
            # Main Training Loop
            batchInd1, batchInd2, batchAtt1, batchAtt2, = f(inds1, i), f(inds2, i), f(att1, i), f(att2, i)
            cLoss, pLoss, nLoss = self.predictAndUpdate(batchInd1, batchAtt1, batchInd2, batchAtt2, f(y, i))

            # Convert Losses into numpy format, instead of TF tensors, for faster np operations
            contrastiveLosses.append(cLoss.numpy())
            pLosses.append(pLoss.numpy())
            nLosses.append(nLoss.numpy())

        return np.mean(contrastiveLosses), np.mean(pLosses), np.mean(nLosses)
