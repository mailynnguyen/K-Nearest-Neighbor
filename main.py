import math
import copy

# GOAL: Nearest Neighbor Classifier
#       1) Forward Selection Search
#       2) Backward Elimination Search
# Datasets have only 2 classes

# Checks the accuracy of the set of features using nearest neighbor classifier
# NOTE: current_set elements range from 1-6
def leave_one_out_cross_validation(list_of_instances, num_instances, current_set, feature_to_add):
    
    num_correctly_classified = 0
    for i in range(num_instances):
        list_of_instance = list_of_instances[i].split()
        object_to_classify = list_of_instance[1:]
        label_object_to_classify = int(float(list_of_instance[0]))
        # print("Looping over i, at the", i + 1, "lcoation")
        # print(f"The {i + 1}th object is in class {int(float(line_list[0]))}")

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')
        for k in range(num_instances):
            if k != i:
                # print(f"Ask if {i + 1} is nearest neighbor with {k + 1}")
                # Euclidean distance: straight line distance btw two points in multi-dimensional space
                # sqrt of sum of sqrd differences of features
                line_list_compare = list_of_instances[k].split()
                object_to_compare = line_list_compare[1:]
                label_object_to_compare = int(float(line_list_compare[0]))

                distance = 0
                for feature in current_set:
                    distance = distance + (float(object_to_classify[feature - 1]) - float(object_to_compare[feature - 1])) ** 2
                distance = distance + (float(object_to_classify[feature_to_add - 1]) - float(object_to_compare[feature_to_add - 1])) ** 2
                # distance = math.sqrt(sum(subtract_and_square(object_to_classify, object_to_compare)))
                distance = math.sqrt(distance)
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k + 1
                    nearest_neighbor_label = label_object_to_compare

        if label_object_to_classify == nearest_neighbor_label:
            num_correctly_classified = num_correctly_classified + 1

        # print(f"Object {i + 1} is class {label_object_to_classify}")
        # print(f"Its nearest neighbor is {nearest_neighbor_location} which is in class {nearest_neighbor_label}")

    accuracy = num_correctly_classified / num_instances
    # print(accuracy)
    return accuracy

    



# Default Rate = size(most common class) / size(dataset)

# Searches through tree to find the most accurate set of features
# Starts with zero features in current set, and slowly adds and checks features
def forward_selection_search(data_objects, num_instances, num_features):

    best_set_of_features = []
    best_set_of_features_accuracy = 0 # maybe sub 0 for default rate
    current_set_of_features = []

    print("Beginning search.")
    print()
    for i in range(1, num_features + 1):
        # print(f"On the {i}th level of the search tree")
        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0

        for k in range(1, num_features + 1):
            if k not in current_set_of_features:

                accuracy = leave_one_out_cross_validation(data_objects, num_instances, current_set_of_features, k)
                
                # if-else for trace printing
                if current_set_of_features:
                    print(f"\tUsing feature(s) {{{",".join(str(feature) for feature in current_set_of_features)},{k}}} accuracy is {accuracy}")
                else:
                    print(f"\tUsing feature(s) {{{k}}} accuracy is {accuracy}")


                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
                
        current_set_of_features.append(feature_to_add_at_this_level)
        
        
        # print(f"On level {i}, I added feature {feature_to_add_at_this_level} to current set")
        print()
        print("{{WARNING: Accuracy has decreased! Continuing search in case of local maxima.}}")
        print(f"Feature set {{{", ".join(str(feature) for feature in current_set_of_features)}}} was best, accuracy is {best_so_far_accuracy}")
        print()

        if best_so_far_accuracy > best_set_of_features_accuracy:
            best_set_of_features_accuracy = best_so_far_accuracy
            best_set_of_features = current_set_of_features.copy()

        # print("best_so_far_accuracy:", best_so_far_accuracy)
        # print("best_set_of_features_accuracy:", best_set_of_features_accuracy)
        # print("best_set_of_features:", best_set_of_features)

    return best_set_of_features, best_set_of_features_accuracy
        
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


    print()
    print(f"This dataset has {num_features} (not including the class attribute), with {num_instances} instances.")
    print()
    print()

    result = []
    if int(user_input_search) == 1:
        result = forward_selection_search(data_objects, num_instances, num_features)
    elif int(user_input_search) == 2:
        result = forward_selection_search(data_objects, num_instances, num_features)
    else:
        print("Invalid Search Choice.")
        return
    
    best_set_of_features, accuracy = result
    print(f"Finished search!! The best feature subset is {{{", ".join(str(feature) for feature in best_set_of_features)}}}, which has an accuracy of {accuracy}")

main()