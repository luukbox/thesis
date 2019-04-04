from pyfsr import NLFSR, FSRFunction
import random
import numpy as np

expression = []

for i in range(4000):
    expression.append(random.randint(0, 3999))

for i in range(3999):
    if random.randint(0, 1) == 1:
        expression.append("+")
    else:
        expression.append("*")

gx = FSRFunction(expression)

nfsr = NLFSR(initstate="random", size=4000, infunc=gx)

print(nfsr.sequence(100).tolist())
