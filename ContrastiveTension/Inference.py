import numpy as np


def applyOutMaskToEmbeddings(embs, mask):
    # Solve potential Padding issues. This can be removed if sufficient precautions are taken
    if (embs.shape[1] > mask.shape[1]):
        mask = np.concatenate(
            [mask, np.zeros((embs.shape[0], embs.shape[1] - mask.shape[1]))], axis=1)
    if (embs.shape[1] < mask.shape[1]):
        mask = mask[:, :embs.shape[1]]

    # Mask the output before calculating the final sentence embedding by taking the mean
    maskedEmbs = embs * np.expand_dims(mask, axis=-1)
    summedEmbs = np.sum(maskedEmbs, axis=1)
    lengths = np.sum(mask, axis=-1, keepdims=True)
    return summedEmbs / lengths


def tensorflowGenerateSentenceEmbeddings(model, tokenizer, texts):
    inData = tokenizer(texts, padding=True, return_tensors='tf')
    inputIds, attentionMask = inData['input_ids'], inData['attention_mask']

    preMaskEmbeddings = model(input_ids=inputIds, attention_mask=attentionMask)[0]
    return applyOutMaskToEmbeddings(preMaskEmbeddings, attentionMask)

def torchGenerateSentenceEmbeddings(model, tokenizer, texts):
    inData = tokenizer(texts, padding=True, return_tensors='pt')
    inputIds, attentionMask = inData['input_ids'], inData['attention_mask']

    f = lambda x: np.array(x.detach())
    preMaskEmbeddings = model(input_ids=inputIds, attention_mask=attentionMask)[0]
    return applyOutMaskToEmbeddings(f(preMaskEmbeddings), f(attentionMask))