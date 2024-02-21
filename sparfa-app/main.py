## mainDriver.cpp

import dataPreparation
import solution
import prediction
import os
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

if __name__ == '__main__':
    # Parameter setting
    K = 6  # number of concepts + 1 (intrinsic difficulty dimension)
    # Low-dimensionality assumption (page 1965), i.e., K << Q, N.
    # Standard techniques like cross-validation can be used to select K. Section 6.3.
    init_mu = 10  # 1e-4  # This represent parameter of l2 regularization
    lbd = 1  # This represents parameter of l1 regularization: lambda, controlling the sparsity level
    init_gamma = 10  # 1e-4  # This represents parameter of Frobenius norm
    I_inner = 10  # number of inner iterations for solving sub problems
    I_outer = 1000  # Number of Iteration of main optimization loop
    # Sparsity assumption (page 1966): Each question should be associated with
    # only a small subset of the concepts in the domain of the course/assessment.
    # In other words, we assume that the matrix W is sparsely populated,
    # i.e., contains mostly zero entries
    init_stepSize = 10  # 4  # The step size in the optimization process
    numOfInitialization = 5  # the number of times we run SPARFA-M


    ############  READ LEARNER RESPONSE Y  #############
    path = os.getcwd()
    csv_file = os.path.join(path, "data\\93per.csv")
    data = pd.read_csv(csv_file)
    data = data.drop(columns=["USER_UUID"])
    print("Learner responses have been loaded.")
    data = data.transpose()  # pay attention to the order of dimensions
    Q = data.shape[0]
    N = data.shape[1]
    observed_response_percentage = 1 - data.isnull().sum().sum() / data.size
    print(f"The data set consists of N = {N} learners answering Q = {Q} questions, with {observed_response_percentage:.1%} of learner responses observed.")
    data = data.fillna(-1).astype(int)
    learner_responses = data.to_numpy()
    ####################################################


    ############  TRAIN TEST DATASET SPLIT  ###############
    X = learner_responses.flatten()  # learner responses 2D to 1D
    y = list(range(learner_responses.size))  # index of X
    res = dict(zip(y, X))
    res_observed = {key: value for key, value in res.items() if value != -1}  # dict comprehension
    y_observed = list(res_observed.keys())
    X_observed = list(res_observed.values())
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X_observed, y_observed, test_size=test_size, random_state=42)
    true_labels = X_test
    Y_train = learner_responses
    for index in y_test:
        i = index // N
        j = index % N
        Y_train[i][j] = -1  # hold out partial observed responses as the test set
    print(f"The training dataset is {observed_response_percentage * (1 - test_size):.1%} of the whole dataset.")
    print(f"The testing dataset is {observed_response_percentage * test_size:.1%} of the whole dataset.")
    print(f"The unobserved response dataset is {1 - observed_response_percentage :.1%} of the whole dataset.")

    ########################### Training process ##########################
    print(f"Start training...")
    global_min = np.inf
    W = np.zeros(Q * K)
    C = np.zeros(K * N)
    for initIterator in range(0, numOfInitialization):
        mu, gamma, stepSize = init_mu, init_gamma, init_stepSize
        solutionObj = solution.Solution(Y_train, Q, N, K)  # object = filename.classname(variables)
        print("Objective function value: ", solutionObj.objectiveFunction(lbd, mu, gamma))
        print("Optimizing...")
        currentIteration = 1
        counter = 1
        # The outer loop is terminated whenever a maximum number of outer iterations is reached,
        # or if the decrease in the objective function of (P)(page 1967) is smaller than a certain threshold
        while currentIteration < I_outer:
            decrease_threshold = 0.0000001
            old_obj_func = solutionObj.objectiveFunction(lbd, mu, gamma)

            for procIterator in range(0, I_inner):
                # update W:
                # for i in range(0, Q):
                #     for k in range(0, K):
                #         solutionObj.updateWij(i, k, lbd, mu, stepSize)
                iW = random.randint(0, Q - 1)
                kW = random.randint(0, K - 1)
                solutionObj.updateWij(iW, kW, lbd, mu, stepSize)

                # update C:
                # for k in range(0, K):
                #     for j in range(0, N):
                #         solutionObj.updateCij(k, j, gamma, stepSize)
                kC = random.randint(0, K - 1)
                jC = random.randint(0, N - 1)
                solutionObj.updateCij(kC, jC, gamma, stepSize)

            new_obj_func = solutionObj.objectiveFunction(lbd, mu, gamma)
            print(f"Iteration {currentIteration}: Objective function value change: {old_obj_func} -> {new_obj_func}")

            if abs(old_obj_func - new_obj_func) < decrease_threshold:
                print("Decrease in the objective function is smaller than the predefined threshold.\n")
                break

            stepSize = stepSize/1.5
            mu = mu/1.5
            gamma = gamma/1.5
            currentIteration += 1

            # for every few outer iterations, if some columns in W contain only zero entries,
            # then we re-initialize them with i.i.d. Gaussian vectors
            if currentIteration % 10 == 0:
                for k in range(0, K):
                    indicator = True
                    for i in range(0, Q):
                        if solutionObj.W[i * K + k] != 0:
                            indicator = False
                    if indicator:
                        for i in range(0, Q):
                            solutionObj.W[i * K + k] = np.random.normal()

            # if currentIteration % 100 == 0:
            #     counter = counter + 1
            #     stepSize = (1/counter) * init_stepSize
            #     mu = (1/counter) * init_mu
            #     gamma = (1/counter) * init_gamma

            # Since the problem (P) is bi-convex in nature, we cannot guarantee that SPARFA-M always
            # converges to a global optimum from an arbitrary starting point.
            # Nevertheless, the use of multiple randomized initialization points and
            # picking the solution with the smallest overall objective function can be used to
            # increase the chance of being in the close vicinity of a global optimum.
            if new_obj_func < global_min:
                global_min = new_obj_func
                W = solutionObj.getW()
                C = solutionObj.getC()
    print(f"The smallest overall objective function value is {global_min}.\n")


######################### Prediction on observed learner responses ######################
# Correct answer likelihood estimation
# Detect associated abstract underlying concepts for each question
# Estimate concept knowledge for each learner
########################################################################
    predictionObj = prediction.Prediction(Y_train, W, C, Q, N, K)
    student_number = 5
    question_numbers = range(0, Q)
    print(f"For Learner {student_number}: ")
    for i in question_numbers:
        print(f"Question {i}: Y = {predictionObj.get_Y(i, student_number)}")
        print(f"Correct answer likelihood: p = {predictionObj.likelihood(i, student_number)}")
        print(f"Underlying concepts: {predictionObj.concepts(i)}")
        print(f"Intrinsic difficulty: {round(predictionObj.get_intrinsic_difficulty()[i], 2)}")
    print(f"Estimated concept knowledge for this learner: {predictionObj.knowledge(student_number)}")


######################## Predicting unobserved learner responses ########################
# prediction accuracy
# confusion matrix
##########################################################################
    pred_labels = []
    for index in y_test:
        i = index // N
        j = index % N
        pred_label = 1 if predictionObj.likelihood(i, j) >= 0.5 else 0
        pred_labels.append(pred_label)
    print(f"The prediction accuracy on the testing dataset is {accuracy_score(true_labels, pred_labels):.2%}.")
    print(f"The confusion matrix:\n {confusion_matrix(true_labels, pred_labels)}.")
    target_names = ['Incorrect', 'Correct']
    print(f"The classification report:\n {classification_report(true_labels, pred_labels, target_names=target_names)}")




























