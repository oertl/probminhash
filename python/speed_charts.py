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
resultFilePrefix = 'performance_test_result_'

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


def drawChart(ax, data, hashSize, mode, isLastRow, isFirstCol):

    r = {}
    algorithms = set()
    for d in data:
        if int(d[2]) != hashSize or d[5] != mode:
            continue
        algorithm = d[0]
        algorithms.add(algorithm)
        dataSize = int(d[3])
        avgCalcTime = float(d[4])
        r.setdefault(dataSize, {}).setdefault(algorithm, []).append(avgCalcTime)

    algorithms = list(algorithms)
    if (mode == "exp(1)"):
        sortedAlgorithms = ["P-MinHash", "ProbMinHash1", "ProbMinHash1a", "ProbMinHash2", "ProbMinHash3", "ProbMinHash3a", "ProbMinHash4", "NonStreamingProbMinHash2", "NonStreamingProbMinHash4"]
        title = r"$m=" + str(hashSize) + r"\quad w(d)\sim\text{Exp}(1)$"
    elif (mode == "pareto(1,0.5)"):
        sortedAlgorithms = ["P-MinHash", "ProbMinHash1", "ProbMinHash1a", "ProbMinHash2", "ProbMinHash3", "ProbMinHash3a", "ProbMinHash4", "NonStreamingProbMinHash2", "NonStreamingProbMinHash4"]
        title = r"$m=" + str(hashSize) + r"\quad w(d)\sim\text{Pareto}(1,0.5)$"
    elif (mode == "pareto(1,2)"):
        sortedAlgorithms = ["P-MinHash", "ProbMinHash1", "ProbMinHash1a", "ProbMinHash2", "ProbMinHash3", "ProbMinHash3a", "ProbMinHash4", "NonStreamingProbMinHash2", "NonStreamingProbMinHash4"]
        title = r"$m=" + str(hashSize) + r"\quad w(d)\sim\text{Pareto}(1,2)$"
    elif (mode == "unweighted"):
        sortedAlgorithms = ["P-MinHash", "ProbMinHash1", "ProbMinHash1a", "ProbMinHash2", "ProbMinHash3", "ProbMinHash3a", "ProbMinHash4", "MinHash", "SuperMinHash", "OPH", "NonStreamingProbMinHash2", "NonStreamingProbMinHash4"]
        title = r"$m=" + str(hashSize) + r"\quad w(d)=1$"
    else:
        assert(False)

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

    ax.set_title(title, fontsize=10)

    for algorithm in algorithms:
        y = [r[dataSize][algorithm] for dataSize in dataSizes]
        ax.plot(dataSizes, y, marker='.', label=algorithm.replace("NonStreamingProbMinHash", "NonStreamingPMH"), color=color_defs.colors[algorithm], linewidth=1)

    handles, labels = ax.get_legend_handles_labels()

    while len(handles) < 12:
        handles.append(plt.Line2D([],[], alpha=0))
        labels.append('')

    order = [1,2,3,4,5, 6, 0, 7,8,9,10,11]

    leg = ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc=2, ncol=2, numpoints=1)

    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor('none')

hashSizes = [256, 1024, 4096]
modes = ["exp(1)", "pareto(1,2)", "pareto(1,0.5)", "unweighted"]

data = readData()

fig, ax = plt.subplots(3, 4, sharex=True, sharey="row")
fig.set_size_inches(16, 9)

for i in range(0, len(hashSizes)):
    for j in range(0, len(modes)):
        drawChart(ax[i][j], data, hashSizes[i], modes[j], i + 1 == len(hashSizes), j == 0)

fig.subplots_adjust(left=0.037, bottom=0.045, right=0.994, top=0.975, wspace=0.05, hspace=0.15)
fig.savefig("paper/speed_charts.pdf", format='pdf', dpi=1200, metadata={'creationDate': None})
plt.close(fig)
