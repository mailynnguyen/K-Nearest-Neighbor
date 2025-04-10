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
                distance = current_set_distances[i][k] + ((object_to_classify[feature_to_add - 1] - object_to_compare[feature_to_add - 1]) ** 2)
                # temp_current_set_distances[i][k] = distance

                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k + 1
                    nearest_neighbor_label = labels[k]

        if label_object_to_classify == nearest_neighbor_label:
            num_correctly_classified = num_correctly_classified + 1

    accuracy = num_correctly_classified / num_instances
    return accuracy




# Cross validation w/ removing a feature
def leave_one_out_cross_validation(labels, features_values, num_instances, current_set, feature_to_remove, current_set_distances, default_rate):
    
    num_correctly_classified = 0

    if len(current_set) == 1:
        return default_rate, current_set_distances
    
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
            
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k + 1
                    nearest_neighbor_label = labels[k]

        if label_object_to_classify == nearest_neighbor_label:
            num_correctly_classified = num_correctly_classified + 1

    accuracy = num_correctly_classified / num_instances
    return accuracy, current_set_distances