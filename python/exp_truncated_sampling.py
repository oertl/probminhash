#######################################
# Copyright (C) 2019-2020 Otmar Ertl. #
# All rights reserved.                #
#######################################

from collections import OrderedDict
import numpy
from numpy import ones, array
from math import exp, expm1, cosh, tanh, sinh, log, erf, sqrt
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import color_defs

plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='cm10')
params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
plt.rcParams.update(params)

rate = log(2)
xMax = 1
numPoints = 1000

def draw_fig1(ax):
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0, xMax])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0","0.5","1"])
    ax.set_yticks([0,exp(-rate),1])
    ax.set_yticklabels(["0",r"$e^{-\lambda}$", "1"])

    xVals = [i/(numPoints-1)*xMax for i in range(0,numPoints)]
    xVals1 = [i/(numPoints-1) for i in range(0,numPoints)]

    ax.plot(xVals, [exp(-rate*x) for x in xVals], color="black", label=r"$\rho:y=e^{-\lambda x}$", linewidth=1, zorder=300) # function

    ax.fill_between(xVals1, [0. for x in xVals1], [exp(-rate) for x in xVals1], facecolor=color_defs.lightgreen, edgeColor=None, label=r"$A_1$", zorder=-300) # rectangle
    ax.fill_between(xVals1, [exp(-rate) for x in xVals1], [exp(-x*rate) for x in xVals1], faceColor=color_defs.lightblue, edgeColor=None, label=r"$A_2$", zorder=-310)

    ax.text(0.2, 0.25, "$A_1$", horizontalalignment='center', verticalalignment='center')
    ax.text(0.2, 0.7, "$A_2$", horizontalalignment='center', verticalalignment='center')

    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([],[], alpha=0))
    labels.append('')
    order = [0,3,1,2]
    leg = ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc=1, ncol=2, numpoints=1)

    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor('none')

def draw_fig2(ax):

    c2 = -log((1+exp(-rate))*0.5)/rate

    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0, xMax])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\tilde{y}$")
    ax.set_xticks([0,c2,0.5, 1])
    ax.set_xticklabels(["0",r"$c_2$","0.5", "1"])
    ax.set_yticks([exp(-rate),(1+exp(-rate))*0.5, 1])
    ax.set_yticklabels(["0","0.5","1"])

    xVals = [i/(numPoints-1)*xMax for i in range(0,numPoints)]
    xVals1 = [i/(numPoints-1) for i in range(0,numPoints)]
    xValsFirstHalf = [0.5*i/(numPoints-1) for i in range(0,numPoints)]
    xValsSecondHalf = [0.5 + 0.5*i/(numPoints-1) for i in range(0,numPoints)]
    xValsC3 = [c2*i/(numPoints-1) for i in range(0,numPoints)]

    ax.fill_between(xVals1, [exp(-rate) for x in xVals1], [min(1 - x *(1-exp(-rate)), (1 + exp(-rate))*0.5) for x in xVals1], facecolor=color_defs.darkorange, edgeColor=None, label=r"$A_4$", zorder=-300) # rectangle
    ax.fill_between(xValsSecondHalf, [1 - x *(1-exp(-rate)) for x in xValsSecondHalf], [(1 + exp(-rate))*0.5 for x in xValsSecondHalf], facecolor=color_defs.lightred, edgeColor=None, label=r"$A_5$", zorder=-310) # rectangle
    ax.fill_between(xValsFirstHalf,  [(exp(-rate)+1)*0.5 for x in xValsFirstHalf], [1 - x *(1-exp(-rate)) for x in xValsFirstHalf],facecolor=color_defs.lightorange, edgeColor=None, label=r"$A_6$", zorder=-320) # rectangle
    ax.fill_between(xValsC3,  [exp(-rate) for x in xValsC3], [ (1 + exp(-rate))*0.5 for x in xValsC3],facecolor=color_defs.lightviolet, edgeColor=None,  zorder=-230, label=r"$A_3$") # rectangle

    ax.plot(xVals, [exp(-rate*x) for x in xVals], color="black", linewidth=1, label=r"$\rho:\tilde{y}=\frac{e^{\lambda (1-x)}-1}{e^\lambda-1}$", zorder=300) # function
    ax.plot(xVals, [1 - rate*x for x in xVals], color="black", linestyle = "dashed", linewidth=1, label=r"$\text{tangent at 0}:\tilde{y}=1-x\frac{\lambda}{1-e^{-\lambda}}$", zorder=290)
    ax.plot(xVals, [exp(-rate) + (1 - x)*rate*exp(-rate) for x in xVals], color="black", linestyle = "dotted", linewidth=1, label=r"$\text{tangent at 1}:\tilde{y}=(1-x)\frac{\lambda}{e^{\lambda}-1}$", zorder=280)

    ax.plot([0.5], [0.5*(1+ exp(-rate))], marker='o', markersize=3, color="black")
    ax.text(0.5, 0.5*(1+ exp(-rate))+0.02, "(0.5,0.5)")

    ax.text(0.07, 0.6, "$A_3$", horizontalalignment='center', verticalalignment='center')
    ax.text(0.47, 0.6, "$A_4$", horizontalalignment='center', verticalalignment='center')
    ax.text(0.92, 0.6, "$A_5$", horizontalalignment='center', verticalalignment='center')
    ax.text(0.07, 0.875, "$A_6$", horizontalalignment='center', verticalalignment='center')

    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([],[], alpha=0))
    labels.append('')
    order = [6,3,4,5,0,1,2,7]

    leg = ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc=(0.02,-0.04), ncol=2, numpoints=1)

    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor('none')


fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(7.62, 3)
draw_fig1(ax1)
draw_fig2(ax2)
fig.subplots_adjust(left=0.062, bottom=0.13, right=0.994, top=0.994, wspace=0.2, hspace=0.0)
fig.savefig('paper/exp_truncated_sampling.pdf', format='pdf', dpi=1200, metadata={'creationDate': None})
plt.close(fig)
