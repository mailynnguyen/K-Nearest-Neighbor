import cross_validation
import copy

# Searches through tree to find the most accurate set of features
# Starts with zero features in current set, and slowly adds and checks features
def forward_selection_search(data_objects, num_instances, num_features, default_rate):

    best_set_of_features = []
    best_set_of_features_accuracy = default_rate 
    current_set_of_features = []

    print("Beginning search.")
    print()
    print("Default Rate:", default_rate)
    print()
    for i in range(1, num_features + 1):
        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0

        for k in range(1, num_features + 1):
            if k not in current_set_of_features:

                accuracy = cross_validation.add_one_in_cross_validation(data_objects, num_instances, current_set_of_features, k)
                
                # if-else for trace printing
                if current_set_of_features:
                    print(f"\tUsing feature(s) {{{",".join(str(feature) for feature in current_set_of_features)},{k}}} accuracy is {accuracy}")
                else:
                    print(f"\tUsing feature(s) {{{k}}} accuracy is {accuracy}")

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
                
        current_set_of_features.append(feature_to_add_at_this_level)
        
        print()
        if best_so_far_accuracy < best_set_of_features_accuracy:
            print("{{WARNING: Accuracy has decreased! Continuing search in case of local maxima.}}")
        print(f"Feature set {{{",".join(str(feature) for feature in current_set_of_features)}}} was best, accuracy is {best_so_far_accuracy}")
        print()

        if best_so_far_accuracy > best_set_of_features_accuracy:
            best_set_of_features_accuracy = best_so_far_accuracy
            best_set_of_features = current_set_of_features.copy()

        # print("best_so_far_accuracy:", best_so_far_accuracy)
        # print("best_set_of_features_accuracy:", best_set_of_features_accuracy)
        # print("best_set_of_features:", best_set_of_features)

    return best_set_of_features, best_set_of_features_accuracy


# Searches through tree to find the most accurate set of features
# Starts with all features in current set, and slowly removes and checks features
def backward_elimination_search(data_objects, num_instances, num_features):
    
    
    best_set_of_features = []
    current_set_of_features = [feature + 1 for feature in range(num_features)] # all features initially
    # accuracy of using entire set
    # using 0, so don't remove any feature and calc accuracy
    best_set_of_features_accuracy = cross_validation.leave_one_out_cross_validation(data_objects, num_instances, current_set_of_features, 0)

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
                accuracy = cross_validation.leave_one_out_cross_validation(data_objects, num_instances, current_set_of_features, k)

                # if-else for trace printing
                if current_set_of_features:
                    print(f"\tUsing feature(s) {{{",".join(str(feature) for feature in current_set_of_features if feature != k)}}} accuracy is {accuracy}")
                else:
                    print(f"\tUsing feature(s) {{{current_set_of_features}}} accuracy is {accuracy}") # ?

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_remove_at_this_level = k

        current_set_of_features.remove(feature_to_remove_at_this_level)
        print()
        if best_so_far_accuracy < best_set_of_features_accuracy:
            print("{{WARNING: Accuracy has decreased! Continuing search in case of local maxima.}}")
        print(f"Feature set {{{",".join(str(feature) for feature in current_set_of_features)}}} was best, accuracy is {best_so_far_accuracy}")
        print()

        if best_so_far_accuracy > best_set_of_features_accuracy:
            best_set_of_features_accuracy = best_so_far_accuracy
            best_set_of_features = current_set_of_features.copy()

    return best_set_of_features, best_set_of_features_accuracy