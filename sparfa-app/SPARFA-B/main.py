import sampling_utility
import bayesian_inference
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

if __name__ == '__main__':
    # Parameter setting (hyperparameters)
    K = 5  # number of concepts + 1 (intrinsic difficulty dimension)
    # Low-dimensionality assumption (page 1965), i.e., K << Q, N.
    # Standard techniques like cross-validation can be used to select K. Section 6.3.
    alph = 1
    bet = 2
    e = 3
    f = 4
    h = 5
    V_0 = 6
    mu_0 = 7
    v_mu = 8

    ############  READ LEARNER RESPONSE Y  #############
    path = os.getcwd()
    parent = os.path.dirname(path)
    csv_file = os.path.join(parent, "data\\93per.csv")
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
    # 7 steps of MCMC sampling
    def train(B, J):
    # This function performs the Gibbs sampling algorithm
    # for posterior distribution estimates for the parameters of interest
    # Inputs:       B: burn-in, number of first iterations to ensure stochastic convergence
    #               J: required sample size (take autocorrelation into consideration)
    # Outputs:      Wchain, Cchain, Muchain: vectors of J/10 values of matrices respectively

        total_number_of_iterations = B + J
        Wchain = []
        Rchain = []
        Cchain = []
        Muchain = []
        samplerObj = sampling_utility.Sampler(Y_train, Q, N, K, alph, bet, e, f, h, V_0, mu_0, v_mu)  # object = filename.classname(variables)
        for iterator in range(total_number_of_iterations):
            samplerObj.sample_Z()
            samplerObj.sample_Mu()
            samplerObj.sample_C()
            samplerObj.sample_V()
            samplerObj.sample_W()
            samplerObj.sample_Lamda()
            samplerObj.sample_r()
            if iterator > B & iterator % 10 == 0:  # After burn-in, take every 10-th values to mitigate autocorrelation
                Wchain.append(samplerObj.get_W())
                Rchain.append(samplerObj.get_R())
                Cchain.append(samplerObj.get_C())
                Muchain.append(samplerObj.get_mu())

        # point estimation (posterior mean) for W, C and mu
        Muarray = np.array(Muchain)
        Mu_estimate = np.mean(Muarray, axis=0)
        Carray = np.array(Cchain)
        C_estimate = np.mean(Carray, axis=0)
        Rarray = np.array(Rchain)
        R_estimate = np.mean(Rarray, axis=0)
        Warray = np.array(Wchain)
        W_estimate = np.mean(Warray, axis=0)
        for i in range(W_estimate.shape[0]):  # generate a sparse W by examining the posterior mean of the inclusion statistics contained in R.
            for k in range(W_estimate.shape[1]):
                if R_estimate[i][k] < 0.01:
                    W_estimate[i][k] = 0

        return W_estimate, C_estimate, Mu_estimate

    print("Start training...")
    W, C, mu = train(100, 10000)

######################### Inference on observed learner responses ######################
# Correct answer likelihood estimation
# Detect associated abstract underlying concepts for each question
# Estimate concept knowledge for each learner
########################################################################
    inferenceObj = bayesian_inference.Inference(Y_train, W, C, Q, N, K)
    student_number = 5
    question_numbers = range(Q)
    print(f"For Learner {student_number}: ")
    for i in question_numbers:
        print(f"Question {i}: Y = {inferenceObj.get_Y(i, student_number)}")
        print(f"Correct answer likelihood: p = {inferenceObj.likelihood(i, student_number)}")
        print(f"Underlying concepts: {inferenceObj.concepts(i)}")
        print(f"Intrinsic difficulty: {round(inferenceObj.get_intrinsic_difficulty()[i], 2)}")
    print(f"Estimated concept knowledge for this learner: {inferenceObj.knowledge(student_number)}")


######################## Predicting unobserved learner responses ########################
# prediction accuracy
# confusion matrix
##########################################################################
    pred_labels = []
    for index in y_test:
        i = index // N
        j = index % N
        pred_label = 1 if inferenceObj.likelihood(i, j) >= 0.5 else 0
        pred_labels.append(pred_label)
    print(f"The prediction accuracy on the testing dataset is {accuracy_score(true_labels, pred_labels):.2%}.")
    print(f"The confusion matrix:\n {confusion_matrix(true_labels, pred_labels)}.")
    target_names = ['Incorrect', 'Correct']
    print(f"The classification report:\n {classification_report(true_labels, pred_labels, target_names=target_names)}")
