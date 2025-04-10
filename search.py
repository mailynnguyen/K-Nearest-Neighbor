import numpy
import cross_validation

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
    print(f"Default Rate: {default_rate * 100:.1f}%")
    print()
    for i in range(1, num_features + 1):

        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0

        for k in range(1, num_features + 1):

            if k not in current_set_of_features:

                accuracy = cross_validation.add_one_in_cross_validation(labels, features_values, num_instances, current_set_of_features, k, current_set_distances,)
                
                # if-else for trace printing
                if current_set_of_features:
                    print(f"\tUsing feature(s) {{{",".join(str(feature) for feature in current_set_of_features)},{k}}} accuracy is {accuracy * 100:.1f}%")
                else:
                    print(f"\tUsing feature(s) {{{k}}} accuracy is {accuracy * 100:.1f}%")

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
                    # new_current_set_distances = temp_current_set_distances.copy()
                    

        current_set_of_features.append(feature_to_add_at_this_level)
        
        # current set distances
        for l in range(num_instances):
                object_to_classify = features_values[l]
                for j in range(num_instances):
                    object_to_compare = features_values[j]

                    distance = current_set_distances[l][j] + (object_to_classify[feature_to_add_at_this_level - 1] - object_to_compare[feature_to_add_at_this_level - 1]) ** 2
                    current_set_distances[l][j] = distance
        # current_set_distances = new_current_set_distances.copy()


        print()
        if best_so_far_accuracy < best_set_of_features_accuracy:
            print("{{WARNING: Accuracy has decreased! Continuing search in case of local maxima.}}")
        print(f"Feature set {{{",".join(str(feature) for feature in current_set_of_features)}}} was best, accuracy is {best_so_far_accuracy * 100:.1f}%")
        print()

        if best_so_far_accuracy > best_set_of_features_accuracy:
            best_set_of_features_accuracy = best_so_far_accuracy
            best_set_of_features = current_set_of_features.copy()

    return best_set_of_features, best_set_of_features_accuracy




# Searches through tree to find the most accurate set of features
# Starts with all features in current set, and slowly removes and checks features
def backward_elimination_search(list_of_instances, num_instances, num_features, default_rate):
    
    data = numpy.array([list(map(float, instance.split())) for instance in list_of_instances])
    labels = data[:, 0].astype(int)
    features_values = data[:, 1:]
    
    best_set_of_features = []
    current_set_of_features = [feature + 1 for feature in range(num_features)] # all features initially

    current_set_distances = [[0] * num_instances for i in range(num_instances)]

    # accuracy of using entire set
    # using 0, so don't remove any feature and calc accuracy
    best_set_of_features_accuracy, current_set_distances = cross_validation.leave_one_out_cross_validation(labels, features_values, num_instances, current_set_of_features, 0, current_set_distances, default_rate)

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
                accuracy, current_set_distances = cross_validation.leave_one_out_cross_validation(labels, features_values, num_instances, current_set_of_features, k, current_set_distances, default_rate)

                # if-else for trace printing
                if current_set_of_features:
                    print(f"\tUsing feature(s) {{{",".join(str(feature) for feature in current_set_of_features if feature != k)}}} accuracy is {accuracy * 100}")
                else:
                    print(f"\tUsing feature(s) {{{current_set_of_features}}} accuracy is {accuracy * 100}") # ?

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
        print(f"Feature set {{{",".join(str(feature) for feature in current_set_of_features)}}} was best, accuracy is {best_so_far_accuracy * 100}")
        print()

        if best_so_far_accuracy > best_set_of_features_accuracy:
            best_set_of_features_accuracy = best_so_far_accuracy
            best_set_of_features = current_set_of_features.copy()

    return best_set_of_features, best_set_of_features_accuracy