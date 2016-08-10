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
ex = np.where( tootoo == [1,2,3] )[0]
print tootoo[0]
tootoo =  np.delete(tootoo,ex, None )
print tootoo








cmd = 'export TPTP=/home/ayatallah/bachelor/TPTP-v6.3.0 ; /home/ayatallah/bachelor/E/PROVER/./classify_problem --tstp-format /home/ayatallah/bachelor/TPTP-v6.3.0/Problems/GRA/GRA034^1.p'
###args = shlex.split()
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()
print(proc[0])
proclist = []
start= False
temp = ""
i = 0
for i in range(len(proc[0])):
    if start is False and proc[0][i] != "(":
        continue
    elif start is False and proc[0][i] =="(":
        start = True
    elif start is True and proc[0][i] != ")" and proc[0][i] != "," and proc[0][i]!=" ":
        temp = temp+ proc[0][i]
    elif start is True and  proc[0][i] == ",":
        if len(proclist)==15 or len(proclist)==16:
            proclist = proclist + [float(temp)]
        else:
            proclist = proclist + [int(temp)]
        temp = ""
    elif start is True and proc[0][i] == ")":
        proclist = proclist + [int(temp)]
        temp = ""
        start = False
for j in range(1,14,1):
    if j == 10 or j == 11 :
        proclist = proclist +[ int(proc[0][i-(14-j)])]
    else:
        proclist = proclist + [proc[0][i - (14 - j)]]


print proclist
print len(proclist)



def getData(name):
    #array = np.genfromtxt(name, delimiter=",", skip_header=62, skip_footer=13764, usecols=(0,1,2,3,4,59,60,61), dtype=None)
    #array = np.genfromtxt(name, delimiter=",", skip_header=62, skip_footer=13764, usecols=(0,1,2,3,4,59,60,61), usemask=True,dtype=None,missing_values="-",filling_values=0)
    #array = np.genfromtxt(name, delimiter=",", skip_header=62, skip_footer=13764, usecols=(0,1,2,3,4,59,60,61), usemask=True,dtype=None,missing_values="-",filling_values=0)
    #array = np.genfromtxt(name, delimiter=" ", skip_header=61, usecols=(0, 1, 2, 3, 4, 59, 60, 61),
    #        names=[problems, status,userTime,failure,preprocessingTime,heuristic,type,equational],
    #        dtype=[('mystring', 'S25'), ('mystring1', 'S25'),('myfloat', 'f8'),('mystring2', 'S25'),('mystring3', 'S25'),('mystring4', 'S25'),
    #               ('mystring5', 'S25'),('mystring6', 'S5')], missing_values=("-","-","-","-","-","-","-","-"),
    #                      filling_values=("0","0",0.0,"0","0","0","0","0"))
    comment = 0
    for line in open(name):
        li = line.strip()
        if li.startswith("#"):
            comment += 1
    print comment
    df= pd.read_csv(name, header=None,delim_whitespace=True,usecols=[0,1,2,3,4,59,60,61],skiprows=comment,
                    names=[problems, status,userTime,failure,preprocessingTime,heuristic,type,equational])
    df[userTime] = df[userTime].convert_objects(convert_numeric=True)
    #df[userTime] = df[userTime].astype('float32')
    df[userTime].fillna(9999.9, inplace=True)
    print len(df)
    return df

def getData2(name):
    #array = np.genfromtxt(name, delimiter=",", skip_header=62, skip_footer=13764, usecols=(0,1,2,3,4,59,60,61), dtype=None)
    #array = np.genfromtxt(name, delimiter=",", skip_header=62, skip_footer=13764, usecols=(0,1,2,3,4,59,60,61), usemask=True,dtype=None,missing_values="-",filling_values=0)
    #array = np.genfromtxt(name, delimiter=",", skip_header=62, skip_footer=13764, usecols=(0,1,2,3,4,59,60,61), usemask=True,dtype=None,missing_values="-",filling_values=0)
    #array = np.genfromtxt(name, delimiter=" ", skip_header=61, usecols=(0, 1, 2, 3, 4, 59, 60, 61),
    #        names=[problems, status,userTime,failure,preprocessingTime,heuristic,type,equational],
    #        dtype=[('mystring', 'S25'), ('mystring1', 'S25'),('myfloat', 'f8'),('mystring2', 'S25'),('mystring3', 'S25'),('mystring4', 'S25'),
    #               ('mystring5', 'S25'),('mystring6', 'S5')], missing_values=("-","-","-","-","-","-","-","-"),
    #                      filling_values=("0","0",0.0,"0","0","0","0","0"))
    comment = 0
    for line in open(name):
        li = line.strip()
        #print li
        if li.startswith("#"):
            comment += 1
    #print comment
    df= pd.read_csv(name,header=None,delimiter=",",skiprows=comment)

    #print df
    return df


def process_categorical():
    df = np.array(getData2("svmInput.csv"))

    tempU = np.where(df[:,23]=='U')
    tempH = np.where(df[:,23]== 'H')
    tempG = np.where(df[:,23]== 'G')
    df[tempU,23] = 0
    df[tempH,23] = 1
    df[tempG,23] = 2

    tempU = np.where(df[:, 24] == 'U')
    tempH = np.where(df[:, 24] == 'H')
    tempG = np.where(df[:, 24] == 'G')
    df[tempU, 24] = 0
    df[tempH, 24] = 1
    df[tempG, 24] = 2

    tempN = np.where(df[:,25]=='N')
    tempS = np.where(df[:,25]== 'S')
    tempP = np.where(df[:,25]== 'P')
    df[tempU,25] = 0
    df[tempH,25] = 1
    df[tempG,25] = 2

    tempF = np.where(df[:,26]=='F')
    tempS = np.where(df[:,26]== 'S')
    tempM = np.where(df[:,26]== 'M')
    df[tempU,26] = 0
    df[tempH,26] = 1
    df[tempG,26] = 2

    tempN = np.where(df[:,27]=='N')
    tempG = np.where(df[:,27]== 'G')
    df[tempN,27] = 0
    df[tempG,27] = 1

    tempS = np.where(df[:,30]=='S')
    tempM = np.where(df[:,30]== 'M')
    tempL = np.where(df[:,30]== 'L')
    df[tempS,30] = 0
    df[tempM,30] = 1
    df[tempL,30] = 2

    tempS = np.where(df[:,31]=='S')
    tempM = np.where(df[:,31]== 'M')
    tempD = np.where(df[:,31]== 'D')
    df[tempS,31] = 0
    df[tempM,31] = 1
    df[tempD,31] = 2
    x = np.array(df[:,31])
    result = np.empty((5,2))
    print result.shape
    print result[:,0]
    return df

def apply_encoding(df):
    m,n = df.shape
    result = np.empty((m, n+8),dtype="S12")

    result[:,0:23] = df[:, 0:23]
    result[:, 23:25] = df[:, 28:30]

    enc = OneHotEncoder()
    enc2 = OneHotEncoder()
    OneHotEncoder(categorical_features='all', dtype=float, handle_unknown='error', n_values='auto', sparse=True)

    enc.fit(df[:,23:28])
    print df[0,23:28]
    result[0, 25:36] = enc.transform([df[0, 23:28]]).toarray()

    for i in range(m):
        result[i, 25:36] = enc.transform([df[i, 23:28]]).toarray()

    #print enc.transform([df[0,23:28]]).toarray()
    #print enc.transform([df[0, 23:28]]).toarray().size

    enc2.fit(df[:,30:32])
    #print df[0,30:32]
    #print enc2.transform([df[0,30:32]]).toarray()
    #print enc2.transform([df[0, 30:32]]).toarray().size
    result[0, 36:41] = enc2.transform([df[0, 30:32]]).toarray()
    #print result[0, :]
    #print len(enc.transform([df[0,23:28]]))
    for i in range(m):
        result[i, 36:41] = enc2.transform([df[i, 30:32]]).toarray()
    return result

reso = process_categorical()
reso2 = apply_encoding(reso)
f = open("enctrialinp.csv", "w")

np.savetxt(
    f,           # file name
    reso,             # formatting, 2 digits in this case
    delimiter=',',          # column delimiter
    newline='\n',          # file footer
    comments='# ',
    fmt="%s")

f2 = open("enctrialout.csv", "w")

np.savetxt(
    f2,           # file name
    reso2,             # formatting, 2 digits in this case
    delimiter=',',          # column delimiter
    newline='\n',          # file footer
    comments='# ',
    fmt="%s")

#df = getData2("svmInput.csv")
#print np.where(df["X"] == "ALG006-1.p")[0]

#for i in range(22):
#    df["f" + str(i + 1)] = df["f" + str(i + 1)].convert_objects(convert_numeric=True)
#    x[:, i] = df["f" + str(i + 1)]
#df["Y"] = df["Y"].convert_objects(convert_numeric=True)
#y = np.array(df["Y"])

#lin_clf = svm.LinearSVC()
#lin_clf.fit(x, y)
#print lin_clf.classes_
#print lin_clf.get_params(deep=True)
print "hi"

#print lin_clf.predict(np.array([[1,2,3,3,41,1,2,1,2,3,3,0,0,0,20,1,0,2,1,5,5,3]]).reshape(1,-1))
print "hi"
#print np.array([66,170,236,307,1639,66,157,66,158,234,222,157,67,157,157,0,1,2,1,6,3,1]).reshape(1,-1)
#print x
#print y


#global labelsNames
#labelsNames = np.array([[]])
#labelsNames=np.append(labelsNames, [[1,2,3]],axis = 1)
#labelsNames=np.append(labelsNames, [[4,5,6]],axis = 0)
#labelsNames=np.append(labelsNames, [[7,8,9]],axis = 0)
#print labelsNames[0]

#my_list = [[1,2,3],[4,5,6],[7,8,9]]
#col_totals = [ sum(x) for x in zip(*my_list) ]
#print col_totals

#elements = [[1,2,3],[4,5,6],[7,8,9]]
#for row in elements:
    # Loop over columns.
#    for column in row:
#        array = column
#        print(column)
#    print("\n")
#thedata = getData("protokoll_G----_0001_FIFO.csv")
#f = open("mydata.csv", "w")

#np.savetxt(
#    f,           # file name
#    thedata,             # formatting, 2 digits in this case
#    delimiter=',',          # column delimiter
#    newline='\n',           # new line character
#    footer='end of file',   # file footer
#    comments='# ',          # character to use for comments
#    header='Data generated by numpy',
#    fmt="%s %s %.5f %s %s %s %s %s")



thedata = getData("heuristics/protokoll_G----_0036_10cw113fifo.csv")
f = open("mydatatrial.csv", "w")
print ((float(thedata[userTime][0])))
np.savetxt(
    f,           # file name
    thedata,             # formatting, 2 digits in this case
    delimiter=',',          # column delimiter
    newline='\n',           # new line character
    footer='end of file',   # file footer
    comments='# ',          # character to use for comments
    header='Data generated by numpy',
    fmt="%s %s %f %s %s %s %s %s")

#thedata2 = getData2("svmInput.csv")
#f2 = open("mydatatrial999.csv", "w")
#np.savetxt(
#    f2,           # file name
#    thedata2,             # formatting, 2 digits in this case
#    delimiter=',',          # column delimiter
#    newline='\n',           # new line character
#    footer='end of file',   # file footer
#    comments='# ',          # character to use for comments
#    header='Data generated by numpy',
#    fmt="%s")

#thedata = getData("heuristics/protokoll_G----_0036_10cw113fifo.csv")
#f = open("mydata.csv", "w")

#np.savetxt(
#    f,           # file name
#    thedata[preprocessingTime],             # formatting, 2 digits in this case
#    delimiter=',',          # column delimiter
#    newline='\n',           # new line character
#    footer='end of file',   # file footer
#    comments='# ',          # character to use for comments
#    header='Data generated by numpy',
#    fmt="%s")

#def getNumericalStatus(status,time):
#    result = np.zeros(len(status), dtype=np.float_)
#    for i in [i for i,x in enumerate(status) if x == "T"]:
#        result[i] = (time[i])*10000000
#    return result


#def getNumericalStatus(status, time):
#    result1 = [0.0]*(len(status))
#    result = np.empty_like(result1)
#    for i in [i for i,x in enumerate(status) if x == "T"]:
 #       result[i] = 1000000000.0 * float(time[i])
 #   return result

#a = getData("protokoll_G----_0001_FIFO.csv")
#print a
#print getNumericalStatus(a["Status"],a["UserTime"])




###proc2 = subprocess.call('export TPTP=/home/ayatallah/bachelor/TPTP-v6.3.0', shell=True)
