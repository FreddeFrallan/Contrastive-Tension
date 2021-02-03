
def getUniqueCaptions(dataset, sortOnSize=True):
    captions = set()
    for s1, s2, _ in dataset:
        captions.add(s1)
        captions.add(s2)
    if sortOnSize:
        return sorted(captions, key=len)
    else:
        return list(captions)


def _readAndLoadSTSBData(name):
    data = []
    with open("STSData/{}".format(name), 'r') as fp:
        for line in fp.readlines():
            genre, filename, year, ids, score, sentence1, sentence2 = line.strip().split('\t')[:7]
            data.append((sentence1, sentence2, float(score)))
    return data


def loadTestData():
    return _readAndLoadSTSBData("sts-test.csv")


def loadDevData():
    return _readAndLoadSTSBData("sts-dev.csv")


def loadTrainData():
    return _readAndLoadSTSBData("sts-train.csv")
