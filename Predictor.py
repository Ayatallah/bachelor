import numpy as np
import os
from time import time
from sklearn import cluster
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
import pandas as pd
import subprocess
from sklearn import svm

class Predictor(object):
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

    def __init__(self, heuristicsDir,clustersno,tptpDir,proverDir,problemsCount):
        self.heuristicsDir = heuristicsDir
        self.clustersno = int(clustersno)
        self.tptpDir = tptpDir
        self.proverDir = proverDir
        self.data = np.array([])
        self.processingData = np.array([])
        self.problemsNameSet = False
        self.problemsNames = np.array([])
        self.heuristicNames = ["Problem"]
        self.problems_features = {}
        self.filesCount = 0
        self.problemsCount = int(problemsCount)
        self.time = 0
        self.labels = []
        self.centroids = []
        self.clustersLengths = []
        self.clusters = []
        self.best_heuristics = []
        self.estimator = svm.LinearSVC()
        self.test = ""


    def getData(self,name):
        comment = 0
        for line in open(name):
            li = line.strip()
            if li.startswith("#"):
                comment += 1
        df = pd.read_csv(name, header=None, delim_whitespace=True, usecols=[0, 1, 2, 3, 4, 59, 60, 61], skiprows=comment,
                         names=[self.problems, self.status, self.userTime, self.failure, self.preprocessingTime,
                                self.heuristic, self.type, self.equational])
        df[self.userTime] = df[self.userTime].convert_objects(convert_numeric=True)
        df[self.userTime] = df[self.userTime].astype('float')
        df[self.userTime].fillna(99999.9, inplace=True)

        if(self.problemsNameSet is False) :
            self.data[:,0] = df[self.problems]
            self.problemsNames  = np.array(df[self.problems])
            self.problemsNameSet = True
        return df

    @staticmethod
    def getNumericalStatus(status, time):
        result = [99999.9]*(len(status))
        for i in [i for i,x in enumerate(time) if x != 99999.9]:
            result[i] = 1.0 * float(time[i])
        return result

    def performanceVectors(self):
        allFiles = os.listdir(self.heuristicsDir)
        self.filesCount = len([i  for i,x in enumerate(allFiles) if x.endswith(".csv")])
        self.processingData = np.empty((self.problemsCount,self.filesCount),dtype="S10")
        self.data = np.empty((self.problemsCount,self.filesCount+1),dtype="S10")
        counter = 1
        counter2 = 0
        for file in allFiles:
            if file.endswith(".csv"):
                self.heuristicNames += [file[16:-4]]
                filetoOpen = self.heuristicsDir+"/"+file
                data = self.getData(filetoOpen)
                self.processingData[:, counter2] = Predictor.getNumericalStatus(data[self.status],data[self.userTime])
                self.data[:, counter] = Predictor.getNumericalStatus(data[self.status],data[self.userTime])
                counter += 1
                counter2 += 1
        #return excludeData(processData)
        #return processData

    def excludeData(self):
        j,k = self.processingData.shape
        print j,k
        solvedbyAllandNone = []
        print self.filesCount
        for i in range(j):
            l = (np.where(self.processingData[i] != '99999.9')[0]).size
            m = (np.where(self.processingData[i] == '99999.9')[0]).size
            if(l == self.filesCount ):
                solvedbyAllandNone = solvedbyAllandNone  + [i]
            if(m == self.filesCount):
                solvedbyAllandNone = solvedbyAllandNone + [i]
        print len(solvedbyAllandNone)
        print j - len(solvedbyAllandNone)
        #Removing the solved by All and None
        self.processingData = np.delete(self.processingData,solvedbyAllandNone,0)
        self.data = np.delete(self.data,solvedbyAllandNone,0)
        #Removing Problems Names Solved by All and None
        #global problemsNames,problemsOnly
        #problemsNames = np.delete(problemsNames,solvedbyAllandNone,0)
        self.problemsnames = np.delete(self.problemsNames,solvedbyAllandNone)
        file = open("processingData.csv", "w")
        np.savetxt(
            file,  # file name
            self.data,  # formatting, 2 digits in this case
            delimiter=',',  # column delimiter
            newline='\n',  # new line character file footer
            comments='#',  # character to use for comments
            fmt="%s")
        #return data

    def cluster_data(self):
        k = self.clustersno
        kmeans = cluster.KMeans(n_clusters=k)
        t0 = time()
        kmeans.fit(self.processingData)
        self.time = time()-t0
        self.labels = kmeans.labels_
        self.centroids = kmeans.cluster_centers_
        #clusters = analyze_data(labels, clustersno)
        #prepare_supervised(labels, clustersno)
        #global best_heuristics
        #best_heuristics = choose_best_heuristics(clusters)
        #return labels,centroids


    def plot_clustering_result(self):
        #print best_heuristics
        k = self.clustersno
        for i in range(k):
            # select only data observations with cluster label == i
            ds = self.processingData[np.where(self.labels == i)]
            # plot the data observations
            dots = plt.plot(ds[:, 0], ds[:, 1], 'o')
            # plot the centroids
            lines = plt.plot(self.centroids[i, 0], self.centroids[i, 1], 'kx')
            # make the centroid x's bigger
            plt.setp(lines, ms=15.0)
            plt.setp(lines, mew=2.0)
            plt.setp(dots, ms=7.0)
            plt.setp(dots, mew=2.0)
        plt.show()

    def analyze_data(self):

        #Get name and heuristics times for problems in each cluster and accummulate them in labelsNames
        labelsNames = []
        k = self.clustersno

        for j in range(k):
            temp = [i for i, x in enumerate(self.labels) if x == j]
            problems = self.processingData[temp]
            labelsNames.append(problems)

        file = open("clusteringOutput.csv", "w")

        # Write Header Row, Heuristics Names
        np.savetxt(
            file,  # file name
            self.heuristicNames,  # formatting, 2 digits in this case
            delimiter=',',  # column delimiter
            newline=',',  # new line character
            fmt="%s")

        #Write each cluster in Clustering output file
        for x in range(len(labelsNames)):
            self.clustersLengths += [len(labelsNames[x])]
            np.savetxt(
                file,  # file name
                labelsNames[x],  # formatting, 2 digits in this case
                delimiter=',',  # column delimiter
                newline='\n',  # new line character
                 footer='End of Cluster'+str(x),  # file footer
                comments='#',  # character to use for comments
                header='Data generated by Clustering, in Cluster'+str(x)+" "+str(len(labelsNames[x]))+" Problems",
                fmt="%s")

        self.clusters = labelsNames

    def choose_best_heuristics(self):
        result =[]
        for i in range(len(self.clusters)):
            cluster = ((self.clusters[i])[:, 1:])
            col_totals = [float((sum(map(float, x)))/13774.0) for x in zip(*cluster)]
            best_index = col_totals.index(min(col_totals))
            best_heuristic = self.heuristicNames[best_index+1]
            result = result + [best_heuristic]
        self.best_heuristics = result

    def elbow_method(self):
        k_range = range(1,450)
        k_means = [cluster.KMeans(n_clusters=k).fit(self.processingData) for k in k_range]
        centroids = [X.cluster_centers_ for X in k_means]
        k_euclid = [cdist(self.processingData, cent, 'euclidean') for cent in centroids]
        dist = [np.min(ke, axis=1) for ke in k_euclid]
        wcss = [sum(d ** 2) for d in dist]
        tss = sum(pdist(self.processingData) ** 2) / self.processingData.shape[0]
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

    @staticmethod
    def get_svmData(name):
        comment = 0
        for line in open(name):
            li = line.strip()
            if li.startswith("#"):
                comment += 1
        df = pd.read_csv(name, header=None, delimiter=",", skiprows=comment)
        return df

    def write_svmInput(self, labelsNames, cluster_number_for_each_problem):

        file = open("svmInput.csv", "w")
        counter = 0

        np.savetxt(
            file,  # file name
            ["#X", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15",
             "f16", "f17", "f18", "f19", "f20", "f21", "f22", "Y"],
            delimiter=',',  # new line character
            newline=',',
            fmt="%s")
        np.savetxt(
            file,  # file name
            [""],  # new line character
            newline='\n',
            fmt="%s")
        # global problems_features
        for i in range(len(labelsNames)):
            cmd = 'export TPTP=' + self.tptpDirectory + ' ; ' + self.proverDirectory + '/./classify_problem --tstp-format ' + self.tptpDirectory + '/Problems/' + \
                  labelsNames[i][0:3] + '/' + labelsNames[i]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()
            features_list = Predictor.process_features(proc)
            self.problems_features[labelsNames[i]] = features_list
            features_list = [labelsNames[i]] + features_list + [int(cluster_number_for_each_problem[i])]
            counter += 1
            np.savetxt(
                file,  # file name
                pd.DataFrame(features_list).T,
                delimiter=',',  # new line character
                newline='\n',
                fmt="%s")

    def rewrite_svmInput(self,cluster_number_for_each_problem):

        df = np.array(Predictor.get_svmData("svmInput.csv"))
        s,r = df.shape
        for i in range(s):
            self.problems_features[df[i][0]] = [df[i][1:23]]
        if self.test != "":
            index =  np.where(df[0] == self.test)[0]
            print len(index)
            print index
            print "khlas keda"
            df = np.delete(df, index, 0)
        m, n = df.shape
        print m
        dict = {}
        for i in range(m):
            dict[df[i][0]] = [df[i][1:]]

        svmInput_tobe = np.empty((m, n), dtype="S15")

        for i in range(m):
            temp = np.array([df[i][0]])
            temp = np.append(temp, dict[df[i][0]][0])
            temp[23] = cluster_number_for_each_problem[i]
            svmInput_tobe[i] = temp

        file = open("svmInput.csv", "w")
        np.savetxt(
            file,  # file name
            ["#X", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15",
             "f16", "f17", "f18", "f19", "f20", "f21", "f22", "Y"],
            delimiter=',',  # new line character
            newline=',',
            fmt="%s")
        np.savetxt(
            file,  # file name
            [""],  # new line character
            newline='\n',
            fmt="%s")
        np.savetxt(
            file,  # file name
            svmInput_tobe,
            delimiter=',',  # new line character
            newline='\n',
            fmt="%s")

    def prepare_supervised(self):
        labelsNames = []
        cluster_number_for_each_problem = []
        k = self.clustersno

        for j in range(k):
            temp = [i for i, x in enumerate(self.labels) if x == j]
            cluster_number_for_each_problem += [j] * len(self.problemsNames[temp])
            labelsNames.append(self.problemsNames[temp])

        labelsNames = np.concatenate(labelsNames).ravel()

        if not (os.path.isfile("svmInput.csv")):
            self.write_svmInput(labelsNames, cluster_number_for_each_problem)
        else:
            self.rewrite_svmInput(cluster_number_for_each_problem)

    def build_estimator(self,df):

        m,n = self.processingData.shape
        x = np.empty((m, 22), dtype="f")
        y = np.array([])

        for i in range(22):
            df[(i + 1)] = df[(i + 1)].convert_objects(convert_numeric=True)
            x[:, i] = df[(i + 1)]

        df[23] = df[23].convert_objects(convert_numeric=True)
        y = np.array(df[23])

        self.estimator.fit(x, y)

    def build_predictor(self):
        self.performanceVectors()
        self.excludeData()
        self.cluster_data()
        self.prepare_supervised()
        self.analyze_data()
        self.choose_best_heuristics()
        svm_input = Predictor.get_svmData("svmInput.csv")
        self.build_estimator(svm_input)

    def test_predictor(self, problem):
        self.performanceVectors()
        self.excludeData()

        index = np.where( self.problemsNames == problem )[0]
        print index,self.problemsNames[np.where( self.problemsNames == problem )]
        self.processingData = np.delete(self.processingData, index, 0)
        self.test = problem
        self.cluster_data()
        self.prepare_supervised()
        self.analyze_data()
        self.choose_best_heuristics()
        svm_input = Predictor.get_svmData("svmInput.csv")
        self.build_estimator(svm_input)
        print "hey"
        #print self.problems_features

        test = self.problems_features[problem]
        print test
        return self.make_prediction(test)


    def make_prediction(self, input_features):
        result = self.estimator.predict(np.array([input_features]).reshape(1, -1))[0]
        return result, self.best_heuristics[result]



x = Predictor("/home/ayatallah/bachelor/bachelor/heuristics", 350, " /home/ayatallah/bachelor/TPTP-v6.3.0",
             "/home/ayatallah/bachelor/E/PROVER", 13774)
print x.test_predictor("ALG006-1.p")
#print x.make_prediction([1, 2, 3, 3, 41, 1, 2, 1, 2, 3, 3, 0, 0, 0, 20, 1, 0, 2, 1, 5, 5, 3])