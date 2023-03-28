###### Your ID ######
# ID1: 208578526
# ID2: 336469358
#####################

# imports 
import numpy as np
import pandas as pd


def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    X = (X - np.mean(X, axis=0)) / (np.amax(X, axis=0) - np.amin(X, axis=0))
    y = (y - np.mean(y, axis=0)) / (np.amax(y, axis=0) - np.amin(y, axis=0))

    return X, y


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """

    X_with_ones = np.column_stack((np.ones(X.shape[0]), X))
    X = X_with_ones
    return X


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    m = X.shape[0]
    div_avg = 1 / (2 * m)
    h = X.dot(theta)  # hypothesis function

    J = div_avg * np.sum((h - y) ** 2)
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration

    m = X.shape[0]  # number of instances

    for i in range(num_iters):

        h = X.dot(theta)  # hypothesis function
        gradient = (1/m) * np.dot(X.T, h-y)

        theta = theta - alpha*gradient  # gradient descent

        cost_value = compute_cost(X, y, theta)
        J_history.append(cost_value)

    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []

    mult = np.matmul(X.T, X)  # X^T * X
    inverse = np.linalg.inv(mult)  # inverse matrix
    pinv = np.matmul(inverse, X.T)
    pinv_theta = np.dot(pinv, y)

    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration
    m = X.shape[0]  # number of instances

    for i in range(num_iters):
        h = X.dot(theta)  # hypothesis function
        gradient = (1/m) * np.dot(X.T, h-y)  # (h(x^(i)) - y^(i)) * x_j^(i)
        theta = theta - alpha * gradient  # gradient descent

        cost_value = compute_cost(X, y, theta)

        if J_history and (J_history[-1] - cost_value) < 1e-8:
            break

        J_history.append(cost_value)
    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """

    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}  # {alpha_value: validation_loss}

    n = X_train.shape[1]  # number of features
    np.random.seed(42)
    theta_rand = np.random.random(n)  # initialize n random theta values in [0,1)
    for alpha in alphas:
        theta, _ = efficient_gradient_descent(X_train, y_train, theta_rand, alpha, iterations)
        val_loss = compute_cost(X_val, y_val, theta)
        alpha_dict[alpha] = val_loss

    return alpha_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    num_features = X_train.shape[1]
    np.random.seed(42)

    while len(selected_features) < 5:
        feature_cost_dict = {}

        # Iterate all features exclude bias column
        for i in range(1, num_features):
            if i not in selected_features:
                selected_features.append(i)

                selected_with_bias = [0] + selected_features
                theta_rand = np.random.random(len(selected_with_bias))
                curr_X_train = X_train[:, selected_with_bias]  # all the selected features columns
                curr_X_val = X_val[:, selected_with_bias]

                curr_theta, curr_J_history = efficient_gradient_descent(
                    curr_X_train,
                    y_train,
                    theta_rand,
                    best_alpha,
                    iterations
                )

                val_loss = compute_cost(curr_X_val, y_val, curr_theta)
                feature_cost_dict[i] = val_loss
                selected_features.remove(i)

        best_feature_index = min(feature_cost_dict, key=feature_cost_dict.get)
        selected_features.append(best_feature_index)

    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly
