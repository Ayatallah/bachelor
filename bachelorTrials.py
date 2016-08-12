import numpy as np
import pandas as pd
import subprocess
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
import shlex
n1 = 'a'
n2 = 'b'
n3 = 'c'
n4 = 'd'
n5 = 'e'
n6 = 'f'
n7 = 'g'
n8 = 'h'
statusCol = 1
userTimeCol = 2


problems = "Problems"
status = "Status"
userTime = "UserTime"
failure = "Failure"
preprocessingTime = "PreprocessingTime"
heuristic = "Heuristic"
type = "Type"
equational = "Equational"
x = np.empty((6755,22), dtype="f")
y = np.array([])

aya = np.empty((2,2), dtype="f")

aya[0][1]=0.0
aya[0][0]=1.0
aya[1][0]=2.0
aya[1][1]=3.0
ahm = aya
print ahm,aya

tootoo = np.array([1,2,3,4])
ex = np.where( tootoo <= 3)[0]
print len(ex)
tootoo =  np.delete(tootoo,ex, None )
print tootoo

tico = [1,2,3,4]
ex = np.where( tico<= 3)[0]
print len(ex)
print ex
