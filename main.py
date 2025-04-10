import cProfile
import search

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
    data_objects = data_file.readlines() # list with each element being a line of text from file (one data object)
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
        result = search.forward_selection_search(data_objects, num_instances, num_features, default_rate)
    elif int(user_input_search) == 2:
        result = search.backward_elimination_search(data_objects, num_instances, num_features, default_rate)
    else:
        print("Invalid Search Choice.")
        return
    
    best_set_of_features, accuracy = result
    print(f"Finished search!! The best feature subset is {{{",".join(str(feature) for feature in best_set_of_features)}}}, which has an accuracy of {accuracy * 100:.1f}%")

# main()

cProfile.run('main()')