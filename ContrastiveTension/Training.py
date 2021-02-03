from ContrastiveTension import ContrastiveTensionModel
import tensorflow as tf
import transformers
import numpy as np


def _setOptimizerWithStepWiseLinearLearningRate(model):
    boundaries, values = [500, 1000, 1500, 2000], [1e-5, 8e-6, 6e-6, 4e-6, 2e-6]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    model.optimizer = tf.optimizers.RMSprop(learning_rate_fn)


def _orderBatchSamples(p_ids, p_att, n_ids, n_att, negativeK):
    inds_1, att_1, inds_2, att_2, labels, negCounter = [], [], [], [], [], 0
    for ind, att in zip(p_ids, p_att):
        # Add the positive sample for both models
        inds_1.append(ind)
        att_1.append(att)
        inds_2.append(ind)
        att_2.append(att)

        # Add Negative Samples for Model-1
        inds_1.extend(n_ids[negCounter:negCounter + negativeK])
        att_1.extend(n_att[negCounter:negCounter + negativeK])
        # Add Negative Samples for Model-2
        inds_2.extend(n_ids[negCounter + negativeK:negCounter + negativeK * 2])
        att_2.extend(n_att[negCounter + negativeK:negCounter + negativeK * 2])

        negCounter += negativeK * 2
        labels.extend([1] + [0] * negativeK)  # Generate fitting labels

    f = lambda x: tf.convert_to_tensor(x, dtype=tf.int32)
    g = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)
    return (f(inds_1), f(att_1), f(inds_2), f(att_2)), g(labels)


def generateTrainingSamples(tokenizer, data, num_batches, negative_k=7, max_sent_len=200):
    pos_sents = [data[i] for i in np.random.randint(0, len(data), num_batches)]
    neg_sents = [data[i] for i in np.random.choice(range(len(data)), num_batches * negative_k * 2)]
    enc_sents = tokenizer.batch_encode_plus(pos_sents + neg_sents, add_special_tokens=True, max_length=max_sent_len,
                                            truncation=True, padding=True)

    ids, att = enc_sents['input_ids'], enc_sents['attention_mask']
    p_ids, p_att = ids[:len(pos_sents)], att[:len(pos_sents)]
    n_ids, n_att = ids[len(pos_sents):], att[len(pos_sents):]

    return _orderBatchSamples(p_ids, p_att, n_ids, n_att, negative_k)


def tensorflowContrastiveTension(modelName, tokenizerName, corpusData, evalFunc=None,
                                 batch_size=16, negative_k=7, fetch_size=500, max_sent_len=75):
    fetch_size = fetch_size * int(batch_size / (negative_k + 1))

    # Create CT Model
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizerName)
    m1 = transformers.TFAutoModel.from_pretrained(modelName)
    m2 = transformers.TFAutoModel.from_pretrained(modelName)
    model = ContrastiveTensionModel.ContrastivTensionModel(m1, m2)
    _setOptimizerWithStepWiseLinearLearningRate(model)

    while (True):
        inData, labels = generateTrainingSamples(tokenizer, corpusData, num_batches=fetch_size,
                                                 negative_k=negative_k,
                                                 max_sent_len=max_sent_len,
                                                 )

        loss, pLoss, nLoss = model.fit(inData, labels, batch_size=batch_size)
        print("Loss: {} pLoss: {}  nLoss: {}".format(loss, pLoss, nLoss))

        if (evalFunc != None):
            eval_1 = evalFunc(model.model1, tokenizer)
            print("Evaluation Scores Model-1:", eval_1)
            eval_2 = evalFunc(model.model2, tokenizer)
            print("Evaluation Scores Model-2:", eval_2)
