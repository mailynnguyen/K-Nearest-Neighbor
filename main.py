import math

# GOAL: Nearest Neighbor Classifier
#       1) Forward Selection Search
#       2) Backward Elimination Search
# Datasets have only 2 classes

# Checks the accuracy of the set of features using nearest neighbor classifier
# NOTE: current_set elements range from 1-6
def leave_one_out_cross_validation(data_objects, data_size, current_set, feature_to_add):
    
    num_correctly_classified = 0
    for i in range(data_size):
        line_list = data_objects[i].split()
        object_to_classify = line_list[1:]
        label_object_to_classify = int(float(line_list[0]))
        # print("Looping over i, at the", i + 1, "lcoation")
        # print(f"The {i + 1}th object is in class {int(float(line_list[0]))}")

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')
        for k in range(data_size):
            if k != i:
                # print(f"Ask if {i + 1} is nearest neighbor with {k + 1}")
                # Euclidean distance: straight line distance btw two points in multi-dimensional space
                # sqrt of sum of sqrd differences of features
                line_list_compare = data_objects[k].split()
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

    accuracy = num_correctly_classified / data_size
    print(accuracy)
    return accuracy

    



# Default Rate = size(most common class) / size(dataset)

# Searches through tree to find the most accurate set of features
# Starts with zero features in current set, and slowly adds and checks features
def forward_selection_search():

    data_file = open('CS170_Small_Data__21.txt', 'r')
    data_objects = data_file.readlines() #list with each element being a line of text from file (one data object)
    data_size = len(data_objects)

    num_features = len(data_objects[0].split()) - 1
    current_set_of_features = []

    for i in range(1, num_features + 1):
        print(f"On the {i}th level of the search tree")
        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0

        for k in range(1, num_features + 1):
            if k not in current_set_of_features:

                print(f"Considering adding the {k} feature")
                accuracy = leave_one_out_cross_validation(data_objects, data_size, current_set_of_features, k)

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
                
        current_set_of_features.append(feature_to_add_at_this_level)
        print(f"On level {i}, I added feature {feature_to_add_at_this_level} to current set")

        

forward_selection_search()
