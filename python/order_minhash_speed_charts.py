#######################################
# Copyright (C) 2019-2020 Otmar Ertl. #
# All rights reserved.                #
#######################################

import os
import csv
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import color_defs

plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='cm10')
params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
plt.rcParams.update(params)

dataDir = 'data/'
resultFilePrefix = 'order_minhash_performance_test_result_'

def readData():
    result = []
    for file in os.listdir(dataDir):
        if file.startswith(resultFilePrefix):
            filename = os.path.join(dataDir, file)
            with open(filename, 'r') as file:
                reader = csv.reader(file, skipinitialspace=True, delimiter=';')
                for r in reader:
                    result.append(r)

    return result


def drawChart(ax, data, hashSize, l, isLastRow, isFirstCol):

    r = {}
    algorithms = set()
    for d in data:
        if int(d[2]) != hashSize or int(d[3]) != l:
            continue
        algorithm = d[0]
        algorithms.add(algorithm)
        dataSize = int(d[4])
        avgCalcTime = float(d[5])
        r.setdefault(dataSize, {}).setdefault(algorithm, []).append(avgCalcTime)

    algorithms = list(algorithms)
    sortedAlgorithms = ["OrderMinHash", "FastOrderMinHash1", "FastOrderMinHash1a", "FastOrderMinHash2"]
    assert(sorted(algorithms) == sorted(sortedAlgorithms))

    algorithms = sortedAlgorithms

    dataSizes = list(r.keys())
    dataSizes.sort()

    for k in r.keys():
        assert(len(r[k]) == len(algorithms))

    
    ax.set_xscale("log", basex=10)
    ax.set_yscale("log", basey=10)

    if isFirstCol:
        ax.set_ylabel(r"calculation time (s)")
    if isLastRow:
        ax.set_xlabel(r"$n$")
    ax.set_title(r"$m=" + str(hashSize) + "\quad l=" + str(l) + "$", fontsize=10)

    for algorithm in algorithms:
        y = [r[dataSize][algorithm] for dataSize in dataSizes]
        ax.plot(dataSizes, y, marker='.', label=r""+algorithm, color=color_defs.colors[algorithm],linewidth=1)

    leg = ax.legend(loc=2, ncol=1, numpoints=1)

    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor('none')

hashSizes = [1024]
lValues = [2, 5]

data = readData()

fig, ax = plt.subplots(1, 2, sharex = True, sharey = "row")
fig.set_size_inches(7.62, 3)
for col in range(0, len(lValues)):
    isFirstCol = col == 0
    isLastRow = True
    drawChart(ax[col], data, hashSizes[0], lValues[col], isLastRow, isFirstCol)

fig.subplots_adjust(left=0.077, bottom=0.125, right=0.994, top=0.94, wspace=0.05, hspace=0.15)
fig.savefig("paper/order_minhash_speed_charts.pdf", format='pdf', dpi=1200, metadata={'creationDate': None})
