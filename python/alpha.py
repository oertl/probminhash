#######################################
# Copyright (C) 2019-2020 Otmar Ertl. #
# All rights reserved.                #
#######################################

from collections import OrderedDict
import numpy
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import color_defs

plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='cm10')
params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
plt.rcParams.update(params)

def calculateFactor(u, m):
    s = 0
    for q in range(1, m):
        s += pow(q / (m - 1), u) * (pow((q+1)/m, u) +
                                    pow((q-1)/m, u) - 2*pow(q/m, u))
    return 1. - s * (m - 1)/(u - 1)


def calculateVariance(sketchSize, a, b, x):

    unionCardinality = a + b + x
    jaccard = x / unionCardinality
    return jaccard * (1 - jaccard) / sketchSize * calculateFactor(unionCardinality, sketchSize)


sketchSizes = OrderedDict(
    [(16, color_defs.darkblue), (64, color_defs.darkgreen), (256, color_defs.darkorange), (1024, color_defs.darkred), (4096, color_defs.darkviolet)])

unionCardinalities = [1*pow(1.02, x) for x in range(1, 750)]


fig, ax = plt.subplots(1, 1)
fig.set_size_inches(3.81, 2.2)
ax.set_xscale("log", basex=10)
ax.set_ylim([0.0, 1.05])
ax.set_xlim([1, 1e6])
ax.set_yticks([0,0.5,1])
ax.set_xlabel(r"$u$")
ax.set_ylabel(r"$\alpha(m,u)$")

for m in sketchSizes:

    factors = [calculateFactor(u, m) for u in unionCardinalities]

    ax.plot(unionCardinalities, factors, label=r"$m=" + str(m) + r"$", linewidth=1, color=sketchSizes[m])


leg = ax.legend(loc=(0.66,0), ncol=1, numpoints=1)

leg.get_frame().set_linewidth(0.0)
leg.get_frame().set_facecolor('none')

fig.subplots_adjust(left=0.125, bottom=0.175, right=0.975, top=0.99)
fig.savefig('paper/alpha.pdf', format='pdf', dpi=1200, metadata={'creationDate': None} )
plt.close(fig)
