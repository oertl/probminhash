#######################################
# Copyright (C) 2019-2020 Otmar Ertl. #
# All rights reserved.                #
#######################################

from scipy.stats import ks_2samp
import random

significanceLevel = 0.01
numAlternativeAlgorithms = 3
random.seed(0)

def extractValues(s):
    data = [int(x) for x in s.split(" ")]
    ret = []
    for i in range(0, len(data)):
        for _ in range(0, data[i]):
            ret.append(i + random.uniform(0,1)) # add random jitter to resolve ties
    return ret


with open("data/orderMinhashEquivalenceTest.txt") as f:
    content = f.readlines()


assert(len(content) % (1 + numAlternativeAlgorithms) == 0)

numPatterns = len(content) // (1 + numAlternativeAlgorithms)

for i in range(0, numPatterns):

    referenceValues = extractValues(content[(numAlternativeAlgorithms+1)*i+0])

    # check equivalence of FastOrderMinHash1 and FastOrderMinHash1a
    for x,y in zip(content[(numAlternativeAlgorithms+1)*i+1], content[(numAlternativeAlgorithms+1)*i+2]):
        assert(x==y)

    for k in range(0, numAlternativeAlgorithms):
        alternativeValues = extractValues(content[(numAlternativeAlgorithms+1)*i+1 + k])

        _, pValue = ks_2samp(referenceValues, alternativeValues)

        print(i)
        print(pValue)
        assert(pValue > significanceLevel)
