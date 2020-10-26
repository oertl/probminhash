#######################################
# Copyright (C) 2019-2020 Otmar Ertl. #
# All rights reserved.                #
#######################################

from scipy.special import erfinv
import csv
from math import sqrt
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import color_defs
from collections import OrderedDict

plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='cm10')
params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
plt.rcParams.update(params)

dataFile = 'data/error_test.csv'
errorTableFileName = 'paper/error_table.txt'

JwIdx = 0
JnIdx = 1
JpIdx = 2
algorithmDescriptionIdx = 3
caseDescriptionIdx = 4
numIterationsIdx = 5
hashSizeIdx = 6
caseIdIdx = 7
isUnweightedIdx = 8
dataSizeA = 9
dataSizeB = 10
dataSizeAB = 11
histogramDataIdx = 12

probability = 0.9999
zScoreLimit = sqrt(2.)*erfinv(probability)


def readData():
    result = []
    headers = []
    with open(dataFile, 'r') as file:
        reader = csv.reader(file, skipinitialspace=True, delimiter=';')
        rowCounter = 0
        for r in reader:
            if rowCounter <= 1:
                headers += r
            elif rowCounter % 2 == 0:
                x = []
                x += r
            else:
                x.append([int(y) for y in r])
                result.append(x)
            rowCounter += 1

    assert(headers[JwIdx] == "Jw")
    assert(headers[JnIdx] == "Jn")
    assert(headers[JpIdx] == "Jp")
    assert(headers[algorithmDescriptionIdx] == "algorithmDescription")
    assert(headers[caseDescriptionIdx] == "caseDescription")
    assert(headers[numIterationsIdx] == "numIterations")
    assert(headers[hashSizeIdx] == "hashSize")
    assert(headers[caseIdIdx] == "caseId")
    assert(headers[isUnweightedIdx] == "isUnweighted")
    assert(headers[dataSizeA] == "dataSizeA")
    assert(headers[dataSizeB] == "dataSizeB")
    assert(headers[dataSizeAB] == "dataSizeAB")
    assert(headers[histogramDataIdx] == "histogramEqualSignatureComponents")

    return headers, result


def getEmpiricalIndex(m, histo):
    if histo is None:
        return float('nan')
    s = 0
    c = 0
    for k in range(0, m + 1):
        s += histo[k] * (k / m)
        c += histo[k]
    return s / c


def getEmpiricalMSE(histo, m, J):
    if histo is None:
        return float('nan')
    assert(m + 1 == len(histo))
    s = 0
    c = 0
    for k in range(0, m + 1):
        c += histo[k]
        s += histo[k] * pow(k / m - J, 2)
    return s / c


def getNumIterations(data):
    n = None
    for d in data:
        if n is None:
            n = int(d[numIterationsIdx])
        else:
            assert(n == int(d[numIterationsIdx]))
    return n


def calculateSigma(J, c, m):
    return sqrt((2. + (1. / (J * (1 - J)) - 6) / m) / c)


def getAlgorithmDescriptions(data, caseId):
    result = set()
    for d in data:
        if (d[caseIdIdx] == caseId):
            result.add(d[algorithmDescriptionIdx])

    if(isUnweighted(data, caseId)):
        sortedAlgorithms = ["P-MinHash", "ProbMinHash1", "ProbMinHash1a",
                            "ProbMinHash2", "ProbMinHash3", "ProbMinHash3a", "ProbMinHash4", 
                            "MinHash", "SuperMinHash", "OPH", "NonStreamingProbMinHash2", "NonStreamingProbMinHash4"]
    else:
        sortedAlgorithms = ["P-MinHash", "ProbMinHash1", "ProbMinHash1a",
                            "ProbMinHash2", "ProbMinHash3", "ProbMinHash3a", "ProbMinHash4", "NonStreamingProbMinHash2", "NonStreamingProbMinHash4"]

    assert(sorted(sortedAlgorithms) == sorted(result))

    return sortedAlgorithms


def getCaseDescriptions(data):
    return sorted(set([d[caseDescriptionIdx] for d in data]))

def getCaseIds(data):
    a = list(OrderedDict.fromkeys([d[caseIdIdx] for d in data]))
    order = [0,7,4,2,1,11,5,3,9,10,8,6]
    assert(sorted(set(order)) == [i for i in range(0, len(a))])
    return [a[o] for o in order]

def getHashSizes(data):
    return sorted(set([int(d[hashSizeIdx]) for d in data]))


def getDataItem(data, algorithmDescription, caseId, hashSize):
    for d in data:
        if(
            d[algorithmDescriptionIdx] == algorithmDescription and
            d[caseIdIdx] == caseId and
            int(d[hashSizeIdx]) == hashSize
        ):
            return d


def getJp(data, caseId):
    for d in data:
        if(d[caseIdIdx] == caseId):
            return float(d[JpIdx])


def isUnweighted(data, caseId):
    for d in data:
        if(d[caseIdIdx] == caseId):
            return int(d[isUnweightedIdx]) == 1


def getUnion(data, caseId):
    for d in data:
        if(d[caseIdIdx] == caseId):
            return int(d[dataSizeA]) + int(d[dataSizeB]) - int(d[dataSizeAB])


def getCaseDescription(caseId):
    for d in data:
        if(d[caseIdIdx] == caseId):
            return d[caseDescriptionIdx].replace(r"\symTestCaseIndex", "u")


def calculateFactor(u, m):
    s = 0
    for q in range(1, m):
        s += pow(q / (m - 1), u) * (pow((q+1)/m, u) +
                                    pow((q-1)/m, u) - 2*pow(q/m, u))
    return 1. - s * (m - 1)/(u - 1)

def checkEqualResults(data, algorithmDescription1, algorithmDescription2):
    caseIds = getCaseIds(data)
    hashSizes = getHashSizes(data)
    for caseId in caseIds:
        for hashSize in hashSizes:
            d1 = getDataItem(data, algorithmDescription1, caseId, hashSize)
            d2 = getDataItem(data, algorithmDescription2, caseId, hashSize)
            assert(((d1 is None) and (d2 is None)) or (d1[histogramDataIdx] == d2[histogramDataIdx]))

headers, data = readData()

equivalentAlgorithms = { "ProbMinHash1a":"ProbMinHash1", "ProbMinHash3a": "ProbMinHash3", "NonStreamingProbMinHash2":"ProbMinHash2", "NonStreamingProbMinHash4":"ProbMinHash4"}

for alg1, alg2 in equivalentAlgorithms.items():
    checkEqualResults(data, alg1, alg2)

caseIds = getCaseIds(data)
hashSizes = getHashSizes(data)
numIterations = getNumIterations(data)

numDiscretePoints = 1000
xLowerBound = hashSizes[0]
xUpperBound = hashSizes[-1]
xForZScoreBand = [xLowerBound * pow(xUpperBound / xLowerBound, t / (numDiscretePoints - 1)) for t in range(0, numDiscretePoints)]

fig, axes = plt.subplots(3, 4, sharex=True, sharey=True)
fig.set_size_inches(16, 9)

for row in range(0, 3):
    isLastRow = (row == 2)
    for col in range(0, 4):
    
        isFirstCol = (col == 0)

        caseId = caseIds[row*4+col]
        
        pCaseDescription = getCaseDescription(caseId)

        ax = axes[row][col]
        
        u = getUnion(data, caseId)

        ax.set_xscale("log", basex=2)
        ax.set_xlim([xLowerBound, xUpperBound])
        ax.set_ylim([0, 2])
        if isFirstCol:
            ax.set_ylabel(r"relative empirical MSE")
            #ax.set_ylabel(r"$\text{empirical MSE}/(J_p(1-J_p)/m)$")
        if isLastRow:
            ax.set_xlabel(r"$m$")
        Jp = getJp(data, caseId)

        zScoreUpperLimit = [1. + zScoreLimit * calculateSigma(Jp, numIterations, xx) for xx in xForZScoreBand]
        zScoreLowerLimit = [1. - zScoreLimit * calculateSigma(Jp, numIterations, xx) for xx in xForZScoreBand]

        ax.fill_between(xForZScoreBand, zScoreLowerLimit, zScoreUpperLimit,
                        facecolor='#cccccc', edgecolor=None, label="{:}".format(probability*100) + r"\%")

        ax.set_title(r"$J_p=" + "{:10.3f}".format(Jp) + r"\quad W=" + pCaseDescription[1:], fontsize=10)
        algorithmDescriptions = getAlgorithmDescriptions(data, caseId)

        for algorithmDescription in algorithmDescriptions:
            if (algorithmDescription in equivalentAlgorithms):
                continue
            
            x = []
            y = []
            for hashSize in hashSizes:
                expectedMSE = Jp * (1 - Jp) / hashSize
                d = getDataItem(data, algorithmDescription, caseId, hashSize)
                if d is not None:
                    empiricalMSE = getEmpiricalMSE( d[histogramDataIdx], hashSize, float(Jp))
                    x.append(hashSize)
                    y.append(empiricalMSE / expectedMSE)

            ax.plot(x, y, marker='.', color=color_defs.colors[algorithmDescription], label=algorithmDescription, linewidth=1)

        if isUnweighted(data, caseId):
            xSuperMinHashTheory = [round(x) for x in xForZScoreBand]
            ySuperMinHashTheory = [calculateFactor(u, m) for m in xSuperMinHashTheory]
            ax.plot(xSuperMinHashTheory, ySuperMinHashTheory, label=r"$\alpha(m," + str(u) + r"$)", color="black", linestyle="--", linewidth=0.5)

        handles, labels = ax.get_legend_handles_labels()

        while len(handles) < 10:
            handles.append(plt.Line2D([],[], alpha=0))
            labels.append('')
        
        if isUnweighted(data, caseId):
            order = [0,1,2,3,4,9,8,5,6,7]
        else:
            order = [0,1,2,3,4,5,6,7,8,9]

        leg = ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper left', ncol=2, numpoints=1)

        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_facecolor('none')

fig.subplots_adjust(left=0.034, bottom=0.045, right=0.994, top=0.975, wspace=0.05, hspace=0.15)
fig.savefig("paper/error_charts.pdf", format='pdf', dpi=1200, metadata={'creationDate': None})
plt.close(fig)

exit(0)
