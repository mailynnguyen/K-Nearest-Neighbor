import cProfile

import math
import numpy

# Cross validation w/ adding a feature
def add_one_in_cross_validation(labels, features_values, num_instances, current_set, feature_to_add, current_set_distances):
    
    num_correctly_classified = 0
    
    for i in range(num_instances):

        object_to_classify = features_values[i]
        label_object_to_classify = labels[i]
        
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')
        
        for k in range(num_instances):
            if k != i:
                object_to_compare = features_values[k]

                # Euclidean distance: straight line distance btw two points in multi-dimensional space
                # sqrt of sum of sqrd differences of features
                # distance = 0
                # for feature in current_set:
                #     distance += ((object_to_classify[feature - 1] - object_to_compare[feature - 1]) ** 2)
                distance = current_set_distances[i][k] + ((object_to_classify[feature_to_add - 1] - object_to_compare[feature_to_add - 1]) ** 2)

                # distance = prev_local_best_set_distance + ((object_to_classify[feature_to_add - 1] - object_to_compare[feature_to_add - 1]) ** 2)
                # print("distance_2:", distance2)

                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k + 1
                    nearest_neighbor_label = labels[k]

        if label_object_to_classify == nearest_neighbor_label:
            num_correctly_classified = num_correctly_classified + 1

    accuracy = num_correctly_classified / num_instances
    return accuracy





# Searches through tree to find the most accurate set of features
# Starts with zero features in current set, and slowly adds and checks features
def forward_selection_search(list_of_instances, num_instances, num_features, default_rate):

    best_set_of_features = []
    best_set_of_features_accuracy = default_rate 
    current_set_of_features = []

    data = numpy.array([list(map(float, instance.split())) for instance in list_of_instances])
    labels = data[:, 0].astype(int)
    features_values = data[:, 1:]

    current_set_distances = [[0] * num_instances for i in range(num_instances)]
    
    print("Beginning search.")
    print()
    print("Default Rate:", default_rate)
    print()
    for i in range(1, num_features + 1):

        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0

        for k in range(1, num_features + 1):

            if k not in current_set_of_features:

                accuracy = add_one_in_cross_validation(labels, features_values, num_instances, current_set_of_features, k, current_set_distances)
                
                # if-else for trace printing
                if current_set_of_features:
                    print(f"\tUsing feature(s) {{{",".join(str(feature) for feature in current_set_of_features)},{k}}} accuracy is {accuracy}")
                else:
                    print(f"\tUsing feature(s) {{{k}}} accuracy is {accuracy}")

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        
        current_set_of_features.append(feature_to_add_at_this_level)
        
        # current set distances
        for l in range(num_instances):
                object_to_classify = features_values[l]
                for j in range(num_instances):
                    object_to_compare = features_values[j]

                    distance = current_set_distances[l][j] + (object_to_classify[feature_to_add_at_this_level - 1] - object_to_compare[feature_to_add_at_this_level - 1]) ** 2
                    current_set_distances[l][j] = distance


        print()
        if best_so_far_accuracy < best_set_of_features_accuracy:
            print("{{WARNING: Accuracy has decreased! Continuing search in case of local maxima.}}")
        print(f"Feature set {{{",".join(str(feature) for feature in current_set_of_features)}}} was best, accuracy is {best_so_far_accuracy}")
        print()

        if best_so_far_accuracy > best_set_of_features_accuracy:
            best_set_of_features_accuracy = best_so_far_accuracy
            best_set_of_features = current_set_of_features.copy()

    return best_set_of_features, best_set_of_features_accuracy



# Cross validation w/ removing a feature
def leave_one_out_cross_validation(labels, features_values, num_instances, current_set, feature_to_remove, current_set_distances):
    
    num_correctly_classified = 0

    for i in range(num_instances):

        object_to_classify = features_values[i]
        label_object_to_classify = labels[i]

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        for k in range(num_instances):

            if k != i: # if k != i, compare the ith instance to kth instance

                object_to_compare = features_values[k]

                # Euclidean distance: straight line distance btw two points in multi-dimensional space
                # sqrt of sum of sqrd differences of features
                if feature_to_remove == 0:
                    distance = 0
                    for feature in current_set:
                        distance += (object_to_classify[feature - 1] - object_to_compare[feature - 1]) ** 2
                    current_set_distances[i][k] = distance
                else:
                    distance = current_set_distances[i][k] - ((object_to_classify[feature_to_remove - 1] - object_to_compare[feature_to_remove - 1]) ** 2)
                # current_set_distances[i][k] = distance
            
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k + 1
                    nearest_neighbor_label = labels[k]

        if label_object_to_classify == nearest_neighbor_label:
            num_correctly_classified = num_correctly_classified + 1

    accuracy = num_correctly_classified / num_instances
    return accuracy, current_set_distances



# Searches through tree to find the most accurate set of features
# Starts with all features in current set, and slowly removes and checks features
def backward_elimination_search(list_of_instances, num_instances, num_features):
    
    data = numpy.array([list(map(float, instance.split())) for instance in list_of_instances])
    labels = data[:, 0].astype(int)
    features_values = data[:, 1:]
    
    best_set_of_features = []
    current_set_of_features = [feature + 1 for feature in range(num_features)] # all features initially

    current_set_distances = [[0] * num_instances for i in range(num_instances)]

    # accuracy of using entire set
    # using 0, so don't remove any feature and calc accuracy
    best_set_of_features_accuracy, current_set_distances = leave_one_out_cross_validation(labels, features_values, num_instances, current_set_of_features, 0, current_set_distances)

    
    print(f"Accuracy of {{{",".join(str(feature) for feature in current_set_of_features)}}}: {best_set_of_features_accuracy}")
    print()
    print("Beginning search.")
    print()
    
    for i in range(1, num_features + 1):
        feature_to_remove_at_this_level = []
        best_so_far_accuracy = 0

        for k in range(1, num_features + 1):
            if k in current_set_of_features:
                # accuracy w/o a certain feature (k)
                accuracy, current_set_distances = leave_one_out_cross_validation(labels, features_values, num_instances, current_set_of_features, k, current_set_distances)

                # if-else for trace printing
                if current_set_of_features:
                    print(f"\tUsing feature(s) {{{",".join(str(feature) for feature in current_set_of_features if feature != k)}}} accuracy is {accuracy}")
                else:
                    print(f"\tUsing feature(s) {{{current_set_of_features}}} accuracy is {accuracy}") # ?

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_remove_at_this_level = k

        current_set_of_features.remove(feature_to_remove_at_this_level)

        # current set distances
        for l in range(num_instances):
            object_to_classify = features_values[l]
            for j in range(num_instances):
                object_to_compare = features_values[j]

                distance = current_set_distances[l][j] - (object_to_classify[feature_to_remove_at_this_level - 1] - object_to_compare[feature_to_remove_at_this_level - 1]) ** 2
                current_set_distances[l][j] = distance

        print()
        if best_so_far_accuracy < best_set_of_features_accuracy:
            print("{{WARNING: Accuracy has decreased! Continuing search in case of local maxima.}}")
        print(f"Feature set {{{",".join(str(feature) for feature in current_set_of_features)}}} was best, accuracy is {best_so_far_accuracy}")
        print()

        if best_so_far_accuracy > best_set_of_features_accuracy:
            best_set_of_features_accuracy = best_so_far_accuracy
            best_set_of_features = current_set_of_features.copy()

    return best_set_of_features, best_set_of_features_accuracy



# GOAL: Nearest Neighbor Classifier
#       1) Forward Selection Search
#       2) Backward Elimination Search
# Datasets have only 2 classes

def main():

    print()
    print("Feature(s) Selection Algorithm")
    print("------------------------------")
    user_input_file = input("Type in the name of the file to test: ")
    print()
    print("Type the number of the algorithm you want to run.")
    print()
    print("\t1) Forward Selection")
    print("\t2) Backward Elimination")
    user_input_search = input()
    
    data_file = open(user_input_file, 'r')
    data_objects = data_file.readlines() #list with each element being a line of text from file (one data object)
    num_instances = len(data_objects)
    num_features = len(data_objects[0].split()) - 1

    # Default Rate = size(most common class) / size(dataset)
    CLASS_1 = 1
    CLASS_2 = 2
    num_class_1 = 0
    num_class_2 = 0    
    # 1) find most common class
    for instance in data_objects:
        list_instance = instance.split()
        class_instance = int(float(list_instance[0]))
        # increase corresponding class counter
        if class_instance == CLASS_1:
            num_class_1 = num_class_1 + 1
        elif class_instance == CLASS_2:
            num_class_2 = num_class_2 + 1

    if num_class_2 >= num_class_1:
        size_most_common_class = num_class_2
    else:
        size_most_common_class = num_class_1

    # 2) calculate default rate
    default_rate = size_most_common_class / num_instances

    print()
    print(f"This dataset has {num_features} (not including the class attribute), with {num_instances} instances.")
    print()

    result = []
    if int(user_input_search) == 1:
        result = forward_selection_search(data_objects, num_instances, num_features, default_rate)
    elif int(user_input_search) == 2:
        result = backward_elimination_search(data_objects, num_instances, num_features)
    else:
        print("Invalid Search Choice.")
        return
    
    best_set_of_features, accuracy = result
    print(f"Finished search!! The best feature subset is {{{",".join(str(feature) for feature in best_set_of_features)}}}, which has an accuracy of {accuracy}")

# main()

cProfile.run('main()')