import numpy as np
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
def getData(name):
    #array = np.genfromtxt(name, delimiter=",", skip_header=62, skip_footer=13764, usecols=(0,1,2,3,4,59,60,61), dtype=None)
    #array = np.genfromtxt(name, delimiter=",", skip_header=62, skip_footer=13764, usecols=(0,1,2,3,4,59,60,61), usemask=True,dtype=None,missing_values="-",filling_values=0)
    array = np.genfromtxt(name, delimiter=",", skip_header=62, skip_footer=6164, usecols=(0, 1, 2, 3, 4, 59, 60, 61),
            names=[problems, status,userTime,failure,preprocessingTime,heuristic,type,equational],
            dtype=[('mystring', 'S25'), ('mystring1', 'S25'),('myfloat', 'f8'),('mystring2', 'S25'),('mystring3', 'S25'),('mystring4', 'S25'),
                   ('mystring5', 'S25'),('mystring6', 'S5')], missing_values=("-","-","-","-","-","-","-","-"),
                          filling_values=("0","0",0.0,"0","0","0","0","0"))
    return len(array[problems])
    #return array


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

a = getData("protokoll_G----_0001_FIFO.csv")
print a
#print getNumericalStatus(a["Status"],a["UserTime"])