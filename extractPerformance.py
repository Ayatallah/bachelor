import numpy as np
import sys
import getopt
import os
from time import time
from sklearn import cluster
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
import operator
#number13774
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

heuristicNames = ["Problem"]
problemsNames = np.empty((13774,41), dtype="S15")
problemsNameSet = False
clustersLengths = []
#clusterTrial = False
## Get Data now prints all data with zero equivalent to anything missing

def getData(name):
    array = np.genfromtxt(name, delimiter=" ", skip_header=61,  usecols=(0,1,2,3,4,59,60,61),
            names=[problems, status,userTime,failure,preprocessingTime,heuristic,type,equational],
            dtype=[('mystring', 'S25'), ('mystring1', 'S25'),('myfloat', 'f8'),('mystring2', 'S25'),('mystring3', 'S25'),('mystring4', 'S25'),
                   ('mystring5', 'S25'),('mystring6', 'S5')], missing_values=("-","-","-","-","-","-","-","-"),
                          filling_values=("0","0",0.0,"0","0","0","0","0"))
    global problemsNames,problemsNameSet
    if(problemsNameSet is False) :
        problemsNames[:,0] = array[problems]
        problemsNameSet = True
    return array



def getProblems(name):
    array = np.empty(shape=13774, dtype="S10")
    array[0:] = np.genfromtxt(name, delimiter=",", skip_header=61,usecols=(0), dtype=None)
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



def getNumericalStatus(status, time):
    result = [99999.9]*(len(status))
    for i in [i for i,x in enumerate(status) if x == "T"]:
        result[i] = 1.0 * float(time[i])
    return result

def performanceVectors(heuristicsDir):
    allFiles = os.listdir(heuristicsDir)
    #f = os.listdir(heuristicsDir)[0]
    #print f[16:32]
    filesCount = len([i  for i,x in enumerate(allFiles) if x.endswith(".csv")])
    problemsCount = 13774 #for testing few problems
    processData = np.empty((problemsCount,filesCount),dtype="S10")
    counter = 1
    counter2 = 0
    for file in allFiles:
        if file.endswith(".csv"):
            global heuristicNames
            heuristicNames+=[file[16:-4]]
            filetoOpen = heuristicsDir+"/"+file
            data = getData(filetoOpen)
            processData[:, counter2] = getNumericalStatus(data[status],data[userTime])
            problemsNames[:, counter] = getNumericalStatus(data[status],data[userTime])
            counter += 1
            counter2 += 1
    #print heuristicNames
    return excludeData(processData)

def excludeData(data):
    j,k = data.shape
    solvedbyAll = []
    solvedbyNone = []
    #print "J IS",j
    for i in range(j):
        l = (np.where(data[i] != '99999.9')[0]).size
        m = (np.where(data[i] == '99999.9')[0]).size
        if(l == 40 ):
            solvedbyAll = solvedbyAll  + [i]
        if(m == 40):
            solvedbyAll = solvedbyAll + [i]
    #Removing the solved by All
    data = np.delete(data,solvedbyAll,0)
    #Removing the solved by None
    ##data = np.delete(data, solvedbyNone, 0)
    #Removing Problems Names
    global problemsNames
    problemsNames = np.delete(problemsNames,solvedbyAll,0)
    problemsNames = np.delete(problemsNames, solvedbyNone,0)
    #print data.shape
    return data

def analyze_data(labels,data,k):

    #Get name and heuristics times for problems in each cluster and accummulate them in labelsNames
    labelsNames = []
    for j in range(k):
        temp = [i for i, x in enumerate(labels) if x == j]
        problems = problemsNames[temp]
        labelsNames.append(problems)

    file = open("clusteringOutput.csv", "w")

    # Write Header Row, Heuristics Names
    np.savetxt(
        file,  # file name
        heuristicNames,  # formatting, 2 digits in this case
        delimiter=',',  # column delimiter
        newline=',',  # new line character
        fmt="%s")

    #Write each cluster in Clustering output file
    for x in range(len(labelsNames)):
        global  clustersLengths
        clustersLengths += [len(labelsNames[x])]
        np.savetxt(
            file,  # file name
            labelsNames[x],  # formatting, 2 digits in this case
            delimiter=',',  # column delimiter
            newline='\n',  # new line character
            footer='End of Cluster'+str(x),  # file footer
            comments='#',  # character to use for comments
            header='Data generated by Clustering, in Cluster'+str(x)+" "+str(len(labelsNames[x]))+" Problems",
            fmt="%s")
    print clustersLengths
    print sum(clustersLengths)



def elbow_method(data):
    k_range = range(1,100)
    k_means = [cluster.KMeans(n_clusters=k).fit(data) for k in k_range]
    centroids = [X.cluster_centers_ for X in k_means]
    k_euclid = [cdist(data, cent, 'euclidean') for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    wcss = [sum(d ** 2) for d in dist]
    tss = sum(pdist(data) ** 2) / data.shape[0]
    bss = tss - wcss
    # elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k_range, bss / tss * 100, 'b*-')
    ax.set_ylim((0, 100))
    ax.set_xlim((0, 100))
    plt.grid(True)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Percentage of Variance')
    plt.title('Variance vs. k')
    plt.show()

def cluster_data(data,clustersno):
    k = int(clustersno)
    kmeans = cluster.KMeans(n_clusters=k)
    t0 = time();
    kmeans.fit(data)
    print "time is:",(time()-t0)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    analyze_data(labels,data,k)
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
                #elbow_method(data)
                cluster_data(data,clustersno)
            else:
                print "Please enter a valid Directory"


if __name__ == "__main__":
    main(sys.argv[1:])

