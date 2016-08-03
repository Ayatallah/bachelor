import sys
import getopt
import os
import Predictor

def main(argv):
    heuristicDir=""
    tptpDirectory=""
    proverDirectory=""
    clustersno = 0
    problemsCount = 0
    try:
        opts, args = getopt.getopt(argv, "hi:k:t:p:e:", ["ifile="])
    except getopt.GetoptError:
        print 'extractPerformance.py -i <inputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'extractPerformance -i <inputfile>'
            sys.exit()
        elif opt == '-i':
            heuristicDir = arg
        elif opt == '-k':
            clustersno = arg
        elif opt == '-t':
            tptpDirectory = arg
        elif opt == '-e':
            proverDirectory = arg
        elif opt == '-p':
            problemsCount = arg
            if (os.path.isdir(heuristicDir)):
                estimator = Predictor.Predictor(heuristicDir,clustersno,tptpDirectory,proverDirectory,problemsCount)
                estimator.build_predictor()
                print estimator.make_prediction([1, 2, 3, 3, 41, 1, 2, 1, 2, 3, 3, 0, 0, 0, 20, 1, 0, 2, 1, 5, 5, 3])
            else:
                print "Please enter a valid Directory"


if __name__ == "__main__":
    main(sys.argv[1:])
