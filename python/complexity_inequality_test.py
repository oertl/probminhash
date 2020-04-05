#######################################
# Copyright (C) 2019-2020 Otmar Ertl. #
# All rights reserved.                #
#######################################

from numpy import log

# this is a test for the inequality n + m * log(m) * log(n) <= C * (n + m * log^2(m))
# which can also be proven analytically

C = 3/2

for n in range(1, 10000):
    for m in range(1, 10000):

        logM = log(m)
        logN = log(n)
        LHS = n + m * logM * logN
        RHS = (n + m * logM * logM) * C

        assert(LHS <= RHS)
