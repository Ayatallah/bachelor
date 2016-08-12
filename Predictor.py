import numpy as np
import os
from time import time
from sklearn import cluster
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
import pandas as pd
import subprocess
from sklearn import svm
from sklearn.cross_validation import KFold

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
        self.testingData = np.array([])
        self.processingData = np.array([])
        self.problemsNames = np.array([])
        self.labels = []
        self.centroids = []
        self.clusters = []
        self.best_heuristics = []
        self.problems_features = {}

        self.clustersLengths = []
        self.problemsNameSet = False
        self.heuristicNames = ["Problem"]
        self.filesCount = 0
        self.problemsCount = int(problemsCount)
        self.time = 0

        self.estimator = svm.LinearSVC()
        #self.test = []


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
        self.processingData = np.empty((self.problemsCount,self.filesCount),dtype="S25")
        self.data = np.empty((self.problemsCount,self.filesCount+1),dtype="S25")
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

    def excludeData(self,test_index):
        j,k = self.processingData.shape
        solvedbyAllandNone = []
        for i in range(j):
            l = (np.where(self.processingData[i] != '99999.9')[0]).size
            m = (np.where(self.processingData[i] == '99999.9')[0]).size
            if(l == self.filesCount ):
                solvedbyAllandNone = solvedbyAllandNone  + [i]
            if(m == self.filesCount):
                solvedbyAllandNone = solvedbyAllandNone + [i]
        #Removing the solved by All and None
        self.processingData = np.delete(self.processingData,solvedbyAllandNone,0)
        self.processingData = np.delete(self.processingData, test_index, 0)
        self.data = np.delete(self.data,solvedbyAllandNone,0)
        self.problemsNames= np.delete(self.problemsNames,solvedbyAllandNone)
        self.testingData = self.problemsNames[test_index]
        self.problemsNames = np.delete(self.problemsNames, test_index, 0)


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

    @staticmethod
    def process_features(proc):
        proclist = []
        start = False
        temp = ""
        i = 0
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
        for j in range(1, 14, 1):
            if j == 10 or j == 11:
                proclist = proclist + [ int( proc[0][ i - (14 - j) ] ) ]
            else:
                proclist = proclist + [proc[0][i - (14 - j)]]
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
             "f16", "f17", "f18", "f19", "f20", "f21", "f22","f23","f24","f25","f26","f27","f28","f29","f30","f31","f32","f33",
             "f34","f35","Y"],
            delimiter=',',  # new line character
            newline=',',
            fmt="%s")
        np.savetxt(
            file,  # file name
            [""],  # new line character
            newline='\n',
            fmt="%s")

        for i in range(len(labelsNames)):
            cmd = 'export TPTP=' + self.tptpDir + ' ; ' + self.proverDir + '/./classify_problem --tstp-format ' + self.tptpDir + '/Problems/' + labelsNames[i][0:3] + '/' + labelsNames[i]
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

    def rewrite_svmInput(self,labelsNames,cluster_number_for_each_problem):

        df = np.array(Predictor.get_svmData("svmInput_encoded.csv"))
        m,n = df.shape

        for i in range(m):
            self.problems_features[df[i][0]] = [df[i][1:41]]

        dict = {}
        for i in range(m):
            dict[df[i][0]] = [df[i][1:]]

        k=  len(labelsNames)
        svmInput_tobe = np.empty((k, n), dtype="S25")

        for i in range(k):
            temp = np.array([labelsNames[i]])
            temp = np.append(temp, dict[labelsNames[i]][0])
            temp[41] = cluster_number_for_each_problem[i]
            svmInput_tobe[i] = temp

        file = open("svmInputTest.csv", "w")
        np.savetxt(
            file,  # file name
            ["#X", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15",
             "f16", "f17", "f18", "f19", "f20", "f21", "f22","f23","f24","f25","f26","f27","f28","f29","f30","f31",
             "f32","f33","f34","f35","f36","f37","f38","f39","f40","Y"],
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
            self.rewrite_svmInput(labelsNames,cluster_number_for_each_problem)


    def build_estimator(self,df):

        m,n = self.processingData.shape
        x = np.empty((m, 40), dtype="f")
        y = np.array([])

        for i in range(40):
            df[(i + 1)] = df[(i + 1)].convert_objects(convert_numeric=True)
            x[:, i] = df[(i + 1)]

        df[41] = df[41].convert_objects(convert_numeric=True)
        y = np.array(df[41])

        self.estimator.fit(x, y)

    def build_predictor(self, train_index,test_index):
        self.performanceVectors()
        self.excludeData(test_index)
        self.cluster_data()
        self.prepare_supervised()
        self.analyze_data()
        self.choose_best_heuristics()
        svm_input = Predictor.get_svmData("svmInputTest.csv")
        self.build_estimator(svm_input)

    def rank_heuristics(self, problem):
        index = np.where(self.data[:,0]==problem)
        temp = self.data[index,1:][0]
        dict = {}
        result = [problem]
        for i in range(self.filesCount):
            dict[temp[0][i]]=self.heuristicNames[i+1]
        temp = np.sort(temp)

        result = result + [dict[temp[0][0]],dict[temp[0][1]],dict[temp[0][2]],dict[temp[0][3]],dict[temp[0][4]]]
        return result

    def test_predictor(self):
        kf = KFold(6755, n_folds=10, shuffle=True)

        test_count = 1
        for train_index, test_index in kf:

            x = Predictor("/home/ayatallah/bachelor/bachelor/heuristics", 350, "/home/ayatallah/bachelor/TPTP-v6.3.0",
                          "/home/ayatallah/bachelor/E/PROVER", 13774)
            x.build_predictor(train_index,test_index)

            testing_set = np.array(x.data[:,0])
            testing_set = np.delete(testing_set, train_index)
            testing_list = []

            predictions = []
            file = open("testing_result"+str(test_count)+".csv", "w")
            np.savetxt(
                file,  # file name
                ["#Problem", "H1", "H2", "H3", "H4", "H5", "Result_H", "Prediction"],
                delimiter=',',  # new line character
                newline=',',
                fmt="%s")
            np.savetxt(
                file,  # file name
                [""],  # new line character
                newline='\n',
                fmt="%s")

            for i in range(len(testing_set)):
                tempfeatures_input = np.array(x.problems_features[testing_set[i]][0])
                testing_list = testing_list + [tempfeatures_input]
            for i in range(len(testing_list)):

                m,n = x.make_prediction(testing_list[i])
                validation_set = x.rank_heuristics(testing_set[i])
                tempto = np.array(validation_set)
                temp = np.where(tempto == str(n))[0]
                if temp.size == 0:
                    predictions += [False]
                else:
                    predictions += [True]

                validation_set += [n]
                validation_set += [predictions[i]]
                np.savetxt(
                    file,  # file name
                    validation_set,  # formatting, 2 digits in this case
                    delimiter=',',  # column delimiter
                    newline=',',  # new line character file footer
                    comments='#',  # character to use for comments
                    fmt="%s")
                np.savetxt(
                    file,  # file name
                    [""],  # new line character
                    newline='\n',
                    fmt="%s")

            preds = np.array(predictions)
            good = np.where(preds == True)[0]
            print "Good predictions %:", (100.0*(float(good.size)/len(predictions)))
            test_count += 1




    def make_prediction(self, input_features):
        result = self.estimator.predict(np.array([input_features]).reshape(1, -1))[0]
        return result, self.best_heuristics[result]



x = Predictor("/home/ayatallah/bachelor/bachelor/heuristics", 350, "/home/ayatallah/bachelor/TPTP-v6.3.0",
             "/home/ayatallah/bachelor/E/PROVER", 13774)

x.test_predictor()