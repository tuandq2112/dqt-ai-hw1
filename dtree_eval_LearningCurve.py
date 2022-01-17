'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''

import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score

numOfFoldsPerTrial = 10

def evaluatePerformance(numTrials=100):
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    
    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n,d = X.shape

    # create list to hold data
    treeAccuracies = []
    stumpAccuracies = []
    dt3Accuracies = []

    tree_accuracies = np.zeros((10, numTrials*numOfFoldsPerTrial))
    stump_accuracies = np.zeros((10, numTrials*numOfFoldsPerTrial))
    dt3_accuracies = np.zeros((10, numTrials*numOfFoldsPerTrial))

    # perform 100 trials
    for x in range(0, numTrials):
        # shuffle the data
        idx = np.arange(n)
        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # split the data randomly into 10 folds
        folds = []    
        intervalDivider = math.floor(len(X)/numOfFoldsPerTrial)
        for fold in range(0, numOfFoldsPerTrial):
            # designate a new testing range
            Xtest = X[fold * intervalDivider:(fold + 1) * intervalDivider,:]
            ytest = y[fold * intervalDivider:(fold + 1) * intervalDivider,:]
            Xtrain = X[:(fold * intervalDivider),:]
            ytrain = y[:(fold * intervalDivider),:]
            Xtrain = Xtrain.tolist()
            ytrain = ytrain.tolist()

            # complete the training data set so that it contains all
            # data except for the current test fold
            for dataRow in range((fold + 1) * intervalDivider, len(X)):
                Xtrain.append(X[dataRow])
                ytrain.append(y[dataRow])
            
            total_train = len(ytrain)
            total_test = len(ytest)
            for percent in range(10, 101, 10):
                n_train = math.floor(total_train * percent / 100)
                Xtrain_ = Xtrain[:n_train]
                ytrain_ = ytrain[:n_train]
                # print(Xtrain_)
                
                # train the decision tree
                clf = tree.DecisionTreeClassifier()
                clf = clf.fit(Xtrain_, ytrain_)
                
                # train the 1-level decision tree
                oneLevel = tree.DecisionTreeClassifier(max_depth=1)
                oneLevel = oneLevel.fit(Xtrain_, ytrain_)
                
                # train the 3-level decision tree
                threeLevel = tree.DecisionTreeClassifier(max_depth=3)
                threeLevel = threeLevel.fit(Xtrain_, ytrain_)
                
                # output predictions on the remaining data
                y_pred_tree = clf.predict(Xtest)
                y_pred_stump = oneLevel.predict(Xtest)
                y_pred_dt3 = threeLevel.predict(Xtest)
                
                # compute the training accuracy of the model
                tree_acc = accuracy_score(ytest, y_pred_tree)
                stump_acc = accuracy_score(ytest, y_pred_stump)
                dt3_acc = accuracy_score(ytest, y_pred_dt3)
                # save to the list of all accuracies
                if percent == 100:
                    treeAccuracies.append(tree_acc)
                    stumpAccuracies.append(stump_acc)
                    dt3Accuracies.append(dt3_acc)
                
                idx_percent = math.floor(percent / 10 - 1)
                idx_trial_fold = x * numOfFoldsPerTrial + fold
                tree_accuracies[idx_percent][idx_trial_fold] = tree_acc
                stump_accuracies[idx_percent][idx_trial_fold] = stump_acc
                dt3_accuracies[idx_percent][idx_trial_fold] = dt3_acc
    
    meanDecisionTreeAccuracies = np.mean(tree_accuracies, axis=1)
    stddevDecisionTreeAccuracies = np.std(tree_accuracies, axis=1)
    
    meanDecisionStumpAccuracies = np.mean(stump_accuracies, axis=1)
    stddevDecisionStumpAccuracies = np.std(stump_accuracies, axis=1)
    meanDT3Accuracies = np.mean(dt3_accuracies, axis=1)
    stddevDT3Accuracies = np.std(dt3_accuracies, axis=1)

    train_sizes = np.arange(10, 101, 10)
    fig, ax = plt.subplots(3, 1)
    plot_learning_curve(
        ax[0],
        train_sizes,
        meanDecisionTreeAccuracies,
        stddevDecisionTreeAccuracies,
        "Depth-unlimited decision tree"
    )
    plot_learning_curve(
        ax[1],
        train_sizes,
        meanDecisionStumpAccuracies,
        stddevDecisionStumpAccuracies,
        "Decision stump"
    )
    plot_learning_curve(
        ax[2],
        train_sizes,
        meanDT3Accuracies,
        stddevDT3Accuracies,
        "3-level decision tree"
    )
    plt.show()

            
    # Update these statistics based on the results of your experiment
    meanDecisionTreeAccuracy = np.mean(treeAccuracies)
    stddevDecisionTreeAccuracy = np.std(treeAccuracies)
    meanDecisionStumpAccuracy = np.mean(stumpAccuracies)
    stddevDecisionStumpAccuracy = np.std(stumpAccuracies)
    meanDT3Accuracy = np.mean(dt3Accuracies)
    stddevDT3Accuracy = np.std(dt3Accuracies)

    # make certain that the return value matches the API specification
    stats = np.zeros((3,2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy
    return stats


def plot_learning_curve(
    ax,
    train_sizes,
    mean_accuracies,
    stddev_accuracies,
    title: str
):
    """Plot learning curve."""

    ax.grid()
    ax.set_xlabel("Training samples (%)")
    ax.set_ylabel("Accuracy")
    ax.errorbar(
        train_sizes,
        mean_accuracies,
        yerr=stddev_accuracies,
        fmt='-o',
        color='r',
        label="Standard deviation of test accuracies"
    )
    ax.set_title(title)
    ax.set_ylim(bottom=0, top=1.0)
    

# Do not modify from HERE...
if __name__ == "__main__":
    stats = evaluatePerformance(100)
    print("Decision Tree Accuracy = {}, ({})".format(stats[0,0], stats[0,1]))
    print("Decision Stump Accuracy = {}, ({})".format(stats[1,0], stats[1,1]))
    print("3-level Decision Tree = {}, ({})".format(stats[2,0], stats[2,1]))
# ...to HERE.