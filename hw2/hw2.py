import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    labels = data[:, -1]
    _, label_counts = np.unique(labels, return_counts=True)  # We ignore sorted values
    label_probs = label_counts / len(labels)  # is a nparray
    gini = 1 - sum(label_probs ** 2)
    ###########################################################################
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    labels = data[:, -1]
    _, label_counts = np.unique(labels, return_counts=True)
    label_probs = label_counts / len(labels)
    label_log_probs = np.log2(label_probs)
    entropy = - sum(label_probs * label_log_probs)
    ###########################################################################
    return entropy


def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting
              according to the feature values.
    """
    goodness = 0
    groups = {}  # groups[feature_value] = data_subset
    ###########################################################################
    feature_col = data[:, feature]
    feature_values, value_counts = np.unique(feature_col, return_counts=True)
    impurity_before = impurity_func(data)
    value_weights = value_counts / len(data)

    impurities_sum = 0
    split_info = 0
    for i, feature_value in enumerate(feature_values):
        value_weight = value_weights[i]
        value_subset = data[data[:, feature] == feature_value]
        impurities_sum += value_weight * impurity_func(value_subset)
        groups[feature_value] = value_subset
        split_info -= value_weight * np.log2(value_weight)

    goodness = impurity_before - impurities_sum
    if gain_ratio:
        split_info = max(split_info, 1e-9)  # To avoid division by 0 warning
        goodness = goodness / split_info
    ###########################################################################
    return goodness, groups


class DecisionNode:

    def __init__(self, data, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):

        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []
        self.terminal = False  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        unique_labels, counts = np.unique(self.data[:, -1], return_counts=True)

        # Set the prediction to the most common class label
        pred = unique_labels[np.argmax(counts)]
        ###########################################################################
        return pred

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)

    def split(self, impurity_func):

        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        ###########################################################################
        if self.depth >= self.max_depth or self.terminal:
            return

        # Find the best feature
        best_feature = -1
        best_goodness = -1
        best_groups = {}
        for feature_index in range(self.data.shape[1] - 1):
            goodness, groups = goodness_of_split(self.data, feature_index, impurity_func, self.gain_ratio)
            if goodness > best_goodness:
                best_feature = feature_index
                best_goodness = goodness
                best_groups = groups

        self.feature = best_feature

        if best_goodness == 0:
            self.terminal = True
            return
        if self.depth >= self.max_depth:
            return

        has_passed_chi_condition = self.check_chi_condition(best_groups)

        if self.chi == 1:
            # Chi square test is disabled - just spilt
            for feature_val, val_subset in best_groups.items():
                child = DecisionNode(val_subset, best_feature, self.depth + 1, self.chi, self.max_depth,
                                     self.gain_ratio)
                self.add_child(child, feature_val)
                child.split(impurity_func)

        elif has_passed_chi_condition:
            for feature_val, val_subset in best_groups.items():
                child = DecisionNode(val_subset, best_feature, self.depth + 1, self.chi, self.max_depth,
                                     self.gain_ratio)
                self.add_child(child, feature_val)
                child.split(impurity_func)
        ###########################################################################

    def check_chi_condition(self, subset):
        """ A helper method that checks if the split for this feature
         is random or good
         Note that we have only 2 classes = e and p """
        def calc_chi(data, subset):
            chi_square = 0
            e_count = np.count_nonzero(data[:, -1] == 'e')
            e_prob = e_count / len(data)
            p_prob = 1 - e_prob
            for feature_val in subset:
                val_subset = subset[feature_val]
                D_f = len(val_subset)
                p_f = np.count_nonzero(val_subset[:, -1] == 'e')
                n_f = np.count_nonzero(val_subset[:, -1] == 'p')
                E_e = D_f * e_prob
                E_p = D_f * p_prob
                chi_square += (((p_f - E_e) ** 2) / E_e) + (((n_f - E_p) ** 2) / E_p)
            return chi_square

        feature_df = len(subset) - 1  # The number of different values minus 1
        chi_threshold = chi_table.get(feature_df).get(self.chi)
        chi_square = calc_chi(self.data, subset)

        if chi_threshold is None:
            return False

        return feature_df > 0 and chi_square > chi_threshold


def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset.
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = None
    ###########################################################################
    root = DecisionNode(data, -1, 0, chi, max_depth, gain_ratio)
    root.split(impurity)
    ###########################################################################
    return root


def predict(root, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    curr_node = root
    while len(curr_node.children):
        instance_feature_val = instance[curr_node.feature]
        try:
            child_node_index = curr_node.children_values.index(instance_feature_val)  # returns ValueError if not found
            child_node = curr_node.children[child_node_index]
            curr_node = child_node
        except ValueError:
            break

    pred = curr_node.pred
    ###########################################################################
    return pred


def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    ###########################################################################
    correct_predictions = 0
    for instance in dataset:
        prediction = predict(node, instance)
        instance_label = instance[-1]
        correct_predictions += int(prediction == instance_label)
    accuracy = 100 * (correct_predictions / len(dataset))
    ###########################################################################
    return accuracy


def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels

    Output: the training and testing accuracies per max depth
    """
    training = []
    testing = []
    ###########################################################################
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        model_tree = build_tree(X_train, calc_entropy, gain_ratio=True, max_depth=max_depth)
        training.append(calc_accuracy(model_tree, X_train))
        testing.append(calc_accuracy(model_tree, X_test))
        # print(f"DEPTH={max_depth} --> train acc. = {training[-1]} ||| test acc. = {testing[-1]}")
    ###########################################################################
    return training, testing


def chi_pruning(X_train, X_test):

    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels

    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc = []
    depth = []
    ###########################################################################
    for alpha_risk in chi_table[1].keys():
        model_tree = build_tree(X_train, calc_entropy, True, alpha_risk)
        chi_testing_acc.append(calc_accuracy(model_tree, X_test))
        chi_training_acc.append(calc_accuracy(model_tree, X_train))
        depth.append(get_tree_depth(model_tree))
    ###########################################################################
    return chi_training_acc, chi_testing_acc, depth


def get_tree_depth(node):
    """ A helper function for calculating the depth of a tree """
    if len(node.children) == 0:
        return 0
    all_depths = []  # Holds the depth of all the node's children
    for child in node.children:
        curr_depth = get_tree_depth(child)
        all_depths.append(curr_depth)
    return max(all_depths) + 1


def count_nodes(node):
    """
    Count the number of node in a given tree

    Input:
    - node: a node in the decision tree.

    Output: the number of nodes in the tree.
    """
    n_nodes = None
    ###########################################################################
    c = 0
    if len(node.children) == 0:
        return 1
    else:
        for child in node.children:
            c += count_nodes(child)

    n_nodes = c + 1
    ###########################################################################
    return n_nodes






