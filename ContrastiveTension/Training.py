'''
Please note that this code is optimized towards comprehension and not performance.
'''

from ContrastiveTension import ContrastiveTensionModel
import tensorflow as tf
import transformers
import numpy as np


def _setOptimizerWithStepWiseLinearLearningRate(model):
    # Optimizer hyperparameters according to the original CT paper
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


def tensorflowContrastiveTension(model_name, tokenizer_name, corpus_data, evalFunc=None, epochs=10,
                                 batch_size=16, negative_k=7, fetch_size=500, max_sent_len=75):
    '''
        :param model_name: Huggingface model path
        :param tokenizer_name: Huggingface tokenizer path
        :param corpus_data: List of strings
        :param evalFunc: Function expecting a model, tokenizer and a batch size
        :param epochs: (int) number of epochs
        :param batch_size: (int)
        :param negative_k: Number of negative pairs, per positive par. batch_size=16 & negative_k=7 -> 14 neg + 2 pos
        :param fetch_size: (int) number of batches that are included per "epoch"
        :param max_sent_len: (int) truncation length during tokenization
    '''
    fetch_size = fetch_size * int(batch_size / (negative_k + 1))

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    m1 = transformers.TFAutoModel.from_pretrained(model_name, from_pt=True)
    m2 = transformers.TFAutoModel.from_pretrained(model_name, from_pt=True)
    model = ContrastiveTensionModel.ContrastivTensionModel(m1, m2)
    _setOptimizerWithStepWiseLinearLearningRate(model)

    bestEvalScore = 0
    for e in range(epochs):
        inData, labels = generateTrainingSamples(tokenizer, corpus_data, num_batches=fetch_size,
                                                 negative_k=negative_k,
                                                 max_sent_len=max_sent_len,
                                                 )

        loss, pLoss, nLoss = model.fit(inData, labels, batch_size=batch_size)
        print("Loss: {} pLoss: {}  nLoss: {}".format(loss, pLoss, nLoss))

        # Perform Evaluation of the models between Epochs, if we have passed an evaluation function
        if (evalFunc != None):
            eval_1 = evalFunc(model.model1, tokenizer, batch_size)
            print("Evaluation Scores Model-1:", eval_1)
            eval_2 = evalFunc(model.model2, tokenizer, batch_size)
            print("Evaluation Scores Model-2:", eval_2)

            # Save the model which performed best on the evaluation data
            eval_1, eval_2 = eval_1['Spearman'], eval_2['Spearman']
            topEval = (eval_1, m1) if eval_1 > eval_2 else (eval_2, m2)
            if(topEval[0] >= bestEvalScore):
                bestEvalScore = topEval[0]
                print("New Best Eval Score:", bestEvalScore)
                topEval[-1].save_pretrained("Top-CT-Eval")

