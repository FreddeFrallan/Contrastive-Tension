from TrainingData import RandomTrainingData
from STSData import Evaluation
from ContrastiveTension import ContrastiveTensionModel
import STSData.Dataset as stsDataset
import tensorflow as tf
import transformers


def _setStepWiseLinearLearningRate(model):
    boundaries, values = [500, 1000, 1500, 2000], [1e-5, 8e-6, 6e-6, 4e-6, 2e-6]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    model.optimizer = tf.optimizers.RMSprop(learning_rate_fn)


def encodeText(texts, tokenizer):
    ids = tokenizer.batch_encode_plus(texts, add_special_tokens=True, max_length=512, truncation=True)['input_ids']
    return RandomTrainingData.batchEncodeData(ids)


def main():
    modelName = "bert-base-uncased"
    m1 = transformers.TFAutoModel.from_pretrained(modelName)
    m2 = transformers.TFAutoModel.from_pretrained(modelName)
    tokenizer = transformers.AutoTokenizer.from_pretrained(modelName)

    model = ContrastiveTensionModel.ContrastivTensionModel(m1, m2)
    _setStepWiseLinearLearningRate(model)

    maxSentLength, maxWord = 75, tokenizer.vocab_size - 1
    negativeK, batchSize = 7, 16
    fetchSize = 500 * int(batchSize / (negativeK + 1))  # Fetches data for 500 updates

    evalData = stsDataset.loadDevData()
    textEncodeFunc = lambda x: encodeText(x, tokenizer)

    model1Eval, model2Eval = Evaluation.evaluateOnData(model, textEncodeFunc, evalData)
    print("Pre Training Evaluation Scores:", model1Eval)

    while (True):
        inds1, att1, inds2, att2, labels = RandomTrainingData.generateRandomTrainingData(numBatches=fetchSize,
                                                                                         negativeK=negativeK,
                                                                                         maxWord=maxWord,
                                                                                         maxLength=maxSentLength)

        loss, pLoss, nLoss = model.fit((inds1, att1, inds2, att2), labels, batch_size=batchSize)
        print("Loss: {} pLoss: {}  nLoss: {}".format(loss, pLoss, nLoss))

        model1Eval, model2Eval = Evaluation.evaluateOnData(model, textEncodeFunc, evalData)
        print("Evaluation Scores Model-1:", model1Eval)
        print("Evaluation Scores Model-2:", model2Eval)
