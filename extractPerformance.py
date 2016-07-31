import numpy as np
import sys
import getopt
import os
from time import time
from sklearn import cluster
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
import pandas as pd
import subprocess
from sklearn import svm
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

filesCount = 0

problems = "Problems"
status = "Status"
userTime = "UserTime"
failure = "Failure"
preprocessingTime = "PreprocessingTime"
heuristic = "Heuristic"
type = "Type"
equational = "Equational"

heuristicNames = ["Problem"]
problemsNames = np.empty((13774,40), dtype="S15")
problemsOnly = np.array([])
problemsNameSet = False
clustersLengths = []
best_heuristics = []
supervised_xAndFeatures = []
supervised_y = []

def getData(name):
    comment = 0
    for line in open(name):
        li = line.strip()
        if li.startswith("#"):
            comment += 1
    df = pd.read_csv(name, header=None, delim_whitespace=True, usecols=[0, 1, 2, 3, 4, 59, 60, 61], skiprows=comment,
                     names=[problems, status, userTime, failure, preprocessingTime, heuristic, type, equational])
    df[userTime] = df[userTime].convert_objects(convert_numeric=True)
    df[userTime] = df[userTime].astype('float')
    df[userTime].fillna(99999.9, inplace=True)

    global problemsNames,problemsNameSet, problemsOnly
    if(problemsNameSet is False) :
        problemsNames[:,0] = df[problems]
        problemsOnly  = np.array(df[problems])
        print problemsOnly
        problemsNameSet = True
    return df

def getNumericalStatus(status, time):
    result = [99999.9]*(len(status))
    for i in [i for i,x in enumerate(time) if x != 99999.9]:
        result[i] = 1.0 * float(time[i])
    return result

def performanceVectors(heuristicsDir):
    allFiles = os.listdir(heuristicsDir)
    global filesCount
    filesCount = len([i  for i,x in enumerate(allFiles) if x.endswith(".csv")])
    print filesCount
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
    return excludeData(processData)

def excludeData(data):
    j,k = data.shape
    solvedbyAllandNone = []
    for i in range(j):
        l = (np.where(data[i] != '99999.9')[0]).size
        m = (np.where(data[i] == '99999.9')[0]).size
        if(l == filesCount ):
            solvedbyAllandNone = solvedbyAllandNone  + [i]
        if(m == filesCount):
            solvedbyAllandNone = solvedbyAllandNone + [i]
    #Removing the solved by All and None
    data = np.delete(data,solvedbyAllandNone,0)
    #Removing Problems Names Solved by All and None
    global problemsNames,problemsOnly
    problemsNames = np.delete(problemsNames,solvedbyAllandNone,0)
    print problemsOnly.size
    problemsOnly = np.delete(problemsOnly,solvedbyAllandNone)
    return data

def cluster_data(data,clustersno):
    k = int(clustersno)
    kmeans = cluster.KMeans(n_clusters=k)
    t0 = time()
    kmeans.fit(data)
    print "time is:",(time()-t0)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    #clusters = analyze_data(labels, clustersno)
    #prepare_supervised(labels, clustersno)
    #global best_heuristics
    #best_heuristics = choose_best_heuristics(clusters)
    return labels,centroids

def plot_clustering_result(labels, centroids,data,k):
    #print best_heuristics
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
def analyze_data(labels,clusterno):

    #Get name and heuristics times for problems in each cluster and accummulate them in labelsNames
    labelsNames = []
    k = int(clusterno)
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
    return labelsNames


def choose_best_heuristics(clusters):
    result =[]
    for i in range(len(clusters)):
        cluster = ((clusters[i])[:, 1:])
        col_totals = [float((sum(map(float, x)))/13774.0) for x in zip(*cluster)]
        best_index = col_totals.index(min(col_totals))
        best_heuristic = heuristicNames[best_index+1]
        result = result + [best_heuristic]
    return result

def process_features(proc):
    proclist = []
    start = False
    temp = ""
    for i in range(len(proc[0])):
        if start is False and proc[0][i] != "(":
            continue
        elif start is False and proc[0][i] == "(":
            start = True
        elif start is True and proc[0][i] != ")" and proc[0][i] != "," and proc[0][i] != " ":
            temp = temp + proc[0][i]
        elif start is True and proc[0][i] == ",":
            if len(proclist) == 15 or len(proclist) == 16:
                proclist = proclist + [float(temp)]
            else:
                proclist = proclist + [int(temp)]
            temp = ""
        elif start is True and proc[0][i] == ")":
            proclist = proclist + [int(temp)]
            temp = ""
            start = False
    return proclist

def prepare_supervised(labels, clusterno,tptpDirectory,proverDirectory):
    labelsNames = []
    cluster_number_for_each_problem = []
    k = int(clusterno)

    for j in range(k):
        temp = [i for i, x in enumerate(labels) if x == j]
        cluster_number_for_each_problem += [j]*len(problemsOnly[temp])
        labelsNames.append(problemsOnly[temp])

    if not (os.path.isfile("svmInput.csv")):
        file = open("svmInput.csv", "w")
        print "clusters length",len(cluster_number_for_each_problem)
        labelsNames=np.concatenate(labelsNames).ravel()
        #print labelsNames
        countar = 0
        np.savetxt(
            file,  # file name
            ["#X","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","f16","f17","f18","f19","f20","f21","f22","Y"],
            delimiter=',',  # new line character
            newline=',',
            fmt="%s")
        np.savetxt(
            file,  # file name
            [""],  # new line character
            newline='\n',
            fmt="%s")
        for i in range(len(labelsNames)):
            cmd = 'export TPTP='+tptpDirectory+' ; '+proverDirectory+'/./classify_problem --tstp-format '+tptpDirectory+'/Problems/'+labelsNames[i][0:3]+'/'+labelsNames[i]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()
            features_list = process_features(proc)
            features_list = [labelsNames[i]]+features_list+[int(cluster_number_for_each_problem[i])]
            countar += 1
            print countar
            np.savetxt(
                file,  # file name
                pd.DataFrame(features_list).T,
                delimiter=',',  # new line character
                newline='\n',
                fmt="%s")
    else:
        df = get_svmData("svmInput.csv")


def elbow_method(data):

    k_range = range(1,450)
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
    ax.set_xlim((0, 450))
    plt.grid(True)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Percentage of Variance')
    plt.title('Variance vs. k')
    plt.show()

def get_svmData(name):
    comment = 0
    for line in open(name):
        li = line.strip()
        if li.startswith("#"):
            comment += 1
    df= pd.read_csv(name,header=None,delimiter=",",skiprows=comment)
    return df

def get_estimator(df):

    x = np.empty((6755, 22), dtype="f")
    y = np.array([])

    for i in range(22):
        df[(i + 1)] = df[(i + 1)].convert_objects(convert_numeric=True)
        x[:, i] = df[(i + 1)]
    df[23] = df[23].convert_objects(convert_numeric=True)
    y = np.array(df[23])

    lin_clf = svm.LinearSVC()
    lin_clf.fit(x, y)

    return lin_clf

def run_program(inputfile,clustersno,tptpDirectory,proverDirectory):
    k = int(clustersno)

    data = performanceVectors(inputfile)

    labels,centroids = cluster_data(data, clustersno)
    if not (os.path.isfile("svmInput.csv")):
        prepare_supervised(labels, k, tptpDirectory, proverDirectory)
    else:
        print "NOOOOO"

    clusters = analyze_data(labels, clustersno)

    global best_heuristics
    best_heuristics = choose_best_heuristics(clusters)

    svm_input = get_svmData("svmInput.csv")
    estimator = get_estimator(svm_input)

    #plot_clustering_result(labels,centroids,data,k)

    return estimator

def main(argv):
    inputfile=""
    tptpDirectory=""
    proverDirectory=""
    clustersno = 0
    try:
        opts, args = getopt.getopt(argv, "hi:k:t:p:", ["ifile="])
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
        elif opt == '-t':
            tptpDirectory = arg
        elif opt == '-p':
            proverDirectory = arg
            if (os.path.isdir(inputfile)):
                # elbow_method(data)
                estimator = run_program(inputfile,clustersno,tptpDirectory,proverDirectory)
                print estimator.predict(np.array([[1, 2, 3, 3, 41, 1, 2, 1, 2, 3, 3, 0, 0, 0, 20, 1, 0, 2, 1, 5, 5, 3]]).reshape(1, -1))
            else:
                print "Please enter a valid Directory"


if __name__ == "__main__":
    main(sys.argv[1:])

