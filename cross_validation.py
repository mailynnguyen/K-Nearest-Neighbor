import math

# Cross validation w/ adding a feature
def add_one_in_cross_validation(list_of_instances, num_instances, current_set, feature_to_add):
    
    num_correctly_classified = 0
    for i in range(num_instances):
        list_of_instance = list_of_instances[i].split()
        object_to_classify = list_of_instance[1:]
        label_object_to_classify = int(float(list_of_instance[0]))

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')
        for k in range(num_instances):
            if k != i:

                # Euclidean distance: straight line distance btw two points in multi-dimensional space
                # sqrt of sum of sqrd differences of features
                line_list_compare = list_of_instances[k].split()
                object_to_compare = line_list_compare[1:]
                label_object_to_compare = int(float(line_list_compare[0]))

                distance = 0
                for feature in current_set:
                    distance = distance + (float(object_to_classify[feature - 1]) - float(object_to_compare[feature - 1])) ** 2
                distance = distance + (float(object_to_classify[feature_to_add - 1]) - float(object_to_compare[feature_to_add - 1])) ** 2
                distance = math.sqrt(distance)

                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k + 1
                    nearest_neighbor_label = label_object_to_compare

        if label_object_to_classify == nearest_neighbor_label:
            num_correctly_classified = num_correctly_classified + 1

    accuracy = num_correctly_classified / num_instances
    return accuracy


# Cross validation w/ removing a feature
def leave_one_out_cross_validation(list_of_instances, num_instances, current_set, feature_to_remove):
    
    num_correctly_classified = 0

    for i in range(num_instances):

        list_of_instance = list_of_instances[i].split()
        object_to_classify = list_of_instance[1:]
        label_object_to_classify = int(float(list_of_instance[0]))

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        for k in range(num_instances):

            if k != i: # if k != i, compare the ith instance to kth instance

                # Euclidean distance: straight line distance btw two points in multi-dimensional space
                # sqrt of sum of sqrd differences of features
                line_list_compare = list_of_instances[k].split()
                object_to_compare = line_list_compare[1:]
                label_object_to_compare = int(float(line_list_compare[0]))

                distance = 0
                for feature in current_set:
                    if feature != feature_to_remove:
                        distance = distance + (float(object_to_classify[feature - 1]) - float(object_to_compare[feature - 1])) ** 2
                distance = math.sqrt(distance)
                # print("feature to remove:", distance)

                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k + 1
                    nearest_neighbor_label = label_object_to_compare

        if label_object_to_classify == nearest_neighbor_label:
            num_correctly_classified = num_correctly_classified + 1

    accuracy = num_correctly_classified / num_instances
    return accuracy