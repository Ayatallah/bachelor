import numpy as np
import sys
import getopt
import os
from time import time
from sklearn import cluster
import matplotlib.pyplot as plt

#specify columns names and deal with them (Y)

#Globals
problemsCol = 0
statusCol = 1
userTimeCol = 2
failureCol = 3
preprocessingTimeCol = 4
heuristicCol = 59
typeCol = 60
equationalCol = 61


problems = "Problems"
status = "Status"
userTime = "UserTime"
failure = "Failure"
preprocessingTime = "PreprocessingTime"
heuristic = "Heuristic"
type = "Type"
equational = "Equational"

## Get Data now prints all data with zero equivalent to anything missing

def getData(name):
    array = np.genfromtxt(name, delimiter=" ", skip_header=61,skip_footer=6164,  usecols=(0,1,2,3,4,59,60,61),
            names=[problems, status,userTime,failure,preprocessingTime,heuristic,type,equational],
            dtype=[('mystring', 'S25'), ('mystring1', 'S25'),('myfloat', 'f8'),('mystring2', 'S25'),('mystring3', 'S25'),('mystring4', 'S25'),
                   ('mystring5', 'S25'),('mystring6', 'S5')], missing_values=("-","-","-","-","-","-","-","-"),
                          filling_values=("0","0",0.0,"0","0","0","0","0"))
    return array

#print getData("protokoll_G----_0001_FIFO.csv")


def getProblems(name):
    array = np.empty(shape=7610, dtype="S10")
    array[0:] = np.genfromtxt(name, delimiter=",", skip_header=61, skip_footer=13164,usecols=(0), dtype=None)
    return array


def getStatus(name):
    array = np.genfromtxt(name, delimiter=",", skip_header=62, skip_footer=13764, usecols=(1), dtype=None)
    return array


def getUserTime(name):
    array = np.genfromtxt(name, delimiter=",", skip_header=62, skip_footer=13764, usecols=(2), dtype=None)
    return array

def getFailure(name):
    array = np.genfromtxt(name, delimiter=",", skip_header=62, skip_footer=13764, usecols=(3), dtype=None)
    return array

def getPreprocessingTime(name):
    array = np.genfromtxt(name, delimiter=",", skip_header=62, skip_footer=13764, usecols=(4), dtype=None)
    return array

def getHeuristic(name):
    array = np.genfromtxt(name, delimiter=",", skip_header=62, skip_footer=13764, usecols=(59), dtype=None)
    return array

def getType(name):
    array = np.genfromtxt(name, delimiter=",", skip_header=62, skip_footer=13764, usecols=(60), dtype=None)
    return array

def getEquational(name):
    array = np.genfromtxt(name, delimiter=",", skip_header=62, skip_footer=13764, usecols=(61), dtype=None)
    return array



#def getNumericalStatus(status, time):
#    result = [999999.9]*(len(status))
#    for i in [i for i,x in enumerate(status) if x == "T"]:
#        result[i] = 100.0 * float(time[i])
#    return result

##Solved 1 and Not Solved 0

def getNumericalStatus(status):
    result = np.zeros(len(status), dtype=np.int)
    for i in [i for i,x in enumerate(status) if x == "T"]:
        result[i] = int(1)
    return result

def performanceVectors(heuristicsDir):
    file1 = os.listdir(heuristicsDir)[1]
    #print file1
    allFiles = os.listdir(heuristicsDir)
    filesCount = len([i  for i,x in enumerate(allFiles) if x.endswith(".csv")])
    #print filesCount
    problemsCount = 7610 #for testing few problems
    vectors = np.empty((filesCount+1,7611),dtype="S10")
    processData = np.empty((problemsCount,filesCount),dtype="S10")
    counter = 1
    counter2 = 0
    for file in allFiles:
        if file.endswith(".csv"):
            vectors[counter][0] = os.path.basename(file)[16:-4]
            filetoOpen = heuristicsDir+"/"+file
            data = getData(filetoOpen)
            vectors[counter][1:] = data[ status]
            processData[:, counter2] = getNumericalStatus(data[status])
            counter += 1
            counter2 += 1
    return processData

def cluster_data(data,clustersno):
    k = int(clustersno)
    #print k
    kmeans = cluster.KMeans(n_clusters=k)
    t0 = time();
    kmeans.fit(data)
    print "time is:",(time()-t0)
    labels = kmeans.labels_
    #print labels
    centroids = kmeans.cluster_centers_
    #print centroids
    for i in range(k):
        # select only data observations with cluster label == i
        ds = data[np.where(labels == i)]
        # plot the data observations
        dots = plt.plot(ds[:, 0], ds[:, 1], 'o')
        # plot the centroids
        lines = plt.plot(centroids[i, 0], centroids[i, 1], 'kx')
        # make the centroid x's bigger
        plt.setp(lines, ms=15.0)
        plt.setp(lines, mew=2.0)
        plt.setp(dots, ms=7.0)
        plt.setp(dots, mew=2.0)
    plt.show()

def excludeData(data):
    b,n = data.shape
    #print m,n
    solvedbyAll = []
    solvedbyNone = []
    #print np.count_nonzero([[0, 1, 7, 0, 0]])
    #var1 = []
    #var1 = data[0]
    #print var1
    #for i in range(5):
        #print np.count_nonzero([data[i]])
        #print data[i]
    #print var
    #for i in range(m):
        #print np.count_nonzero(data[i])
    for i in range(b):
        l = (np.where(data[i] == '1')[0]).size
        m = (np.where(data[i] == '0')[0]).size
        if(l == 40 ):
            solvedbyAll = solvedbyAll  + [i]
        if(m == 40):
            solvedbyNone = solvedbyNone + [i]
    #print data.shape
    data_solvedbyAll = (np.delete(data,solvedbyAll,0))
    #print data_solvedbyAll.shape
    data_solvedbyAllandNone = (np.delete(data_solvedbyAll,solvedbyNone,0))
    #print data_solvedbyAllandNone.shape
    #print var
    return data_solvedbyAllandNone.shape


def main(argv):
    inputfile=""
    clustersno = 0
    try:
        opts, args = getopt.getopt(argv, "hi:k:", ["ifile="])
    except getopt.GetoptError:
        print 'extractPerformance.py -i <inputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'extractPerformance -i <inputfile>'
            sys.exit()
        elif opt == '-i':
            inputfile = arg
        elif opt == '-k':
            clustersno = arg
            if (os.path.isdir(inputfile)):
                data = performanceVectors(inputfile)
                excludeData(data)
                #print data
                cluster_data(data,clustersno)
            else:
                print "Please enter a valid Directory"


if __name__ == "__main__":
    main(sys.argv[1:])

