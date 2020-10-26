#######################################
# Copyright (C) 2019-2020 Otmar Ertl. #
# All rights reserved.                #
#######################################

import os
import csv
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
from statistics import mean
import operator
import numpy as np
import color_defs
import math

plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='cm10')
params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
plt.rcParams.update(params)

dataDir = 'data/'
dataFilePrefix = 'buffer_size_test_result_'
dataFilePostFix = ".dat"
data = dict()

dataHashSizePairs = []
for file in os.listdir(dataDir):
    if file.startswith(dataFilePrefix):
        dataHashSizePairs.append(file[len(dataFilePrefix):-len(dataFilePostFix)])

hashSizes = []
dataSizes = []

for x in dataHashSizePairs:
    a = x.split("_")
    hashSizes.append(int(a[0]))
    dataSizes.append(int(a[1]))

hashSizes = sorted(set(hashSizes))
dataSizes = sorted(set(dataSizes))

numCycles = None
maxDataSize = None
    
for hashSize in hashSizes:
    for dataSize in dataSizes:
        dataFile = dataDir + dataFilePrefix + str(hashSize) + "_" + str(dataSize) + dataFilePostFix
        result = list()
        with open(dataFile, 'r') as file:
            reader = csv.reader(file, skipinitialspace=True, delimiter=';')
            if numCycles is None:
                numCycles = int(next(reader)[0])
            else:
                assert(int(next(reader)[0]) == numCycles)
            assert(int(next(reader)[0]) == dataSize)
            headers = next(reader)
            for h in headers:
                if not h in data:
                    data[h] = {s : {t : [] for t in dataSizes} for s in hashSizes}
            for r in reader:
                for i in range(0, len(headers)):
                    data[headers[i]][hashSize][dataSize].append(int(r[i]))
                

algorithmMapping = {
    "probMinHash1a" : "ProbMinHash1a",
    "probMinHash3a" : "ProbMinHash3a",
    "probMinHash1aWeightedExp" : "ProbMinHash1a",
    "probMinHash3aWeightedExp" : "ProbMinHash3a",
    "probMinHash1aWeightedPareto0_5" : "ProbMinHash1a",
    "probMinHash3aWeightedPareto0_5" : "ProbMinHash3a",
    "probMinHash1aWeightedPareto2" : "ProbMinHash1a",
    "probMinHash3aWeightedPareto2" : "ProbMinHash3a"
}

harmonicNumbers = [0]
for i in range(1,dataSizes[-1]+1):
    harmonicNumbers.append(1/i + harmonicNumbers[-1])

def harmonic(i):
    return harmonicNumbers[i]

def expectedBufferSizeProbMinHash1a(n,m):

    a = m * harmonic(m)

    b = math.floor(a)

    if n <= b:
        return n
    else:
        return b + (harmonic(n) - harmonic(b))*a

def expectedBufferSizeProbMinHash3aUnweighted(n,m):
    return min(m * harmonic(m), n)

def plot(ax, hashSize, algorithms, data, title, isLastRow, isFirstCol):

    ax.set_xscale("log",basex=10)
    ax.set_yscale("log")
    
    if isFirstCol:
        ax.set_ylabel(r"buffer size")
    
    if isLastRow:
        ax.set_xlabel(r"$n$")

    ax.set_title(title, fontsize=10)

    xVals = range(1, dataSizes[-1] + 1)

    yValsExpectedBufferSizeProbMinHash1aTheory = [expectedBufferSizeProbMinHash1a(x, hashSize) for x in xVals]
    yValsExpectedBufferSizeProbMinHash3aUnweightedTheory = [expectedBufferSizeProbMinHash3aUnweighted(x, hashSize) for x in xVals]

    for algorithmDescription in algorithms:

        means = [mean(x) for x in data[algorithmDescription][hashSize].values()]
        percentiles995 = [np.percentile(x, 99.5) for x in data[algorithmDescription][hashSize].values()]
        percentiles005 = [np.percentile(x, 0.5) for x in data[algorithmDescription][hashSize].values()]

        mappedAlg = algorithmMapping[algorithmDescription] 

        ax.fill_between(dataSizes, percentiles005, percentiles995,
                        facecolor=color_defs.colors[mappedAlg], edgecolor=None, zorder=-100)

        ax.plot(dataSizes, means, label=mappedAlg, linewidth=0.5, linestyle=None,marker='.', markersize=4,color=color_defs.colors[mappedAlg[:-1]],zorder=100)

    ax.plot(xVals, yValsExpectedBufferSizeProbMinHash1aTheory, linewidth=0.5, linestyle="dashdot",color="black",zorder=1000, label = r"$\sum_{i=1}^n \min(1, m H_m / i)$")
    ax.plot(xVals, yValsExpectedBufferSizeProbMinHash3aUnweightedTheory, linewidth=0.5, linestyle="dotted",color="black",zorder=1100, label=r"$\min(n, m H_m)$")

    leg = ax.legend(loc=2, numpoints=1,frameon = 1,framealpha=1)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor('none')


fig, ax = plt.subplots(3, 4, sharex=True, sharey=True)
fig.set_size_inches(16, 9)

for i in range(0, len(hashSizes)):
    m = hashSizes[i]
    xAxisLabel = i == len(hashSizes)-1
    plot(ax[i][0], m, ["probMinHash1aWeightedExp", "probMinHash3aWeightedExp"], data, r"$m = " + str(m) + r"\quad w(d)\sim\text{Exp}(1)$", xAxisLabel, True)
    plot(ax[i][1], m, ["probMinHash1aWeightedPareto2", "probMinHash3aWeightedPareto2"], data, r"$m = " + str(m) + r"\quad w(d)\sim\text{Pareto}(1,2)$", xAxisLabel, False)
    plot(ax[i][2], m, ["probMinHash1aWeightedPareto0_5", "probMinHash3aWeightedPareto0_5"], data, r"$m = " + str(m) + r"\quad w(d)\sim\text{Pareto}(1,0.5)$", xAxisLabel, False)
    plot(ax[i][3], m, ["probMinHash1a", "probMinHash3a"], data, r"$m = " + str(m) + r"\quad w(d)=1$", xAxisLabel, False)

fig.subplots_adjust(left=0.03, bottom=0.045, right=0.994, top=0.975, wspace=0.05, hspace=0.15)
fig.savefig("paper/buffer_size_charts.pdf", format='pdf', dpi=1200, metadata={'creationDate': None})
plt.close(fig)
