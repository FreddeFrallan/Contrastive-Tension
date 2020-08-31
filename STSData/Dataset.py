def _loadUniqueCaptions(dataset, sortOnSize=True):
    captions = {}
    for s1, s2, _ in dataset:
        captions[s1] = 1
        captions[s2] = 1

    if (sortOnSize):
        captions = sorted([(len(c), c) for c in captions], key=lambda x: x[0])
        return [c[1] for c in captions]
    else:
        return list(captions.keys())


def _readAndLoadSTSBData(filePath):
    data = []
    with open(filePath, 'r') as fp:
        for line in fp.readlines():
            genre, filename, year, ids, score, sentence1, sentence2 = line.strip().split('\t')[:7]
            data.append((sentence1, sentence2, float(score)))
    return data


def loadTestData():
    return _readAndLoadSTSBData("STSData/sts-test.csv")


def loadDevData():
    return _readAndLoadSTSBData("STSData/sts-dev.csv")


def loadTrainData():
    return _readAndLoadSTSBData("STSData/sts-train.csv")
