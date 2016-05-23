import numpy as np
import sys
import getopt
import os

##we want to read files once, read the whole file and throw away columns we dont need from a numpy array
#use global variables to be able change whatever u want, for example columns number to add or remove

#Globals
problems = 0
status = 1
userTime = 2
failure = 3
preprocessingTime = 4
heuristic = 59
type = 60
equational = 61

## Get Data now prints all data with zero equivalent to anything missing
def getData(name):
    array = np.genfromtxt(name, delimiter=",", skip_header=62, skip_footer=13744,  dtype=None, missing_values="-",filling_values="0")
    return array

print getData("protokoll_G----_0001_FIFO.csv")
def getProblems(name):
    array = np.empty(shape=10, dtype="S10")
    array[0:] = np.genfromtxt(name, delimiter=",", skip_header=62, skip_footer=13764, usecols=(0), dtype=None)
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

## we need status and time, no need for combination between them.
##give cell with no data, a zero for example,
##first try to cluster, upon success or no success, get sth simple work first then upgrade

#def getNumericalStatus(status, time):
#    result = [999999.9]*(len(status))
#    for i in [i for i,x in enumerate(status) if x == "T"]:
#        result[i] = 100.0 * float(time[i])
#    return result


def performanceVectors(heuristicsDir):
    file1 = os.listdir(heuristicsDir)[1]
    allFiles = os.listdir(heuristicsDir)
    filesCount = len([i  for i,x in enumerate(allFiles) if x.endswith(".csv")])
    vectors = np.empty((filesCount+1,11),dtype="S10")
    vectors[0][0] = "Problem/Heuristic"
    vectors[0][1:] = getProblems(file1)[:]
    counter = 1
    for file in allFiles:
        if file.endswith(".csv"):
            vectors[counter][0] = os.path.basename(file)[16:-4]
            data = getData(file)
            #vectors[counter][1:] = getNumericalStatus(data[:, status], data[:, userTime])[:]
            vectors[counter][1:] = data[:, status]
            counter += 1
    return vectors


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hi:", ["ifile="])
    except getopt.GetoptError:
        print 'extractPerformance.py -i <inputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'extractPerformance -i <inputfile>'
            sys.exit()
        elif opt == '-i':
            inputfile = arg
            if (os.path.isdir(inputfile)):
                print performanceVectors(inputfile)
            else:
                print "Please enter a valid Directory"


if __name__ == "__main__":
    main(sys.argv[1:])