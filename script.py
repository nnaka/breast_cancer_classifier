import matplotlib
import sklearn

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def main():
    # part 1
    breast_cancer_data = load_breast_cancer()

    # part 2
    print breast_cancer_data.data[0]
    print breast_cancer_data.feature_names

    # part 3
    print breast_cancer_data.target
    print breast_cancer_data.target_names

    # part 5 and 6
    training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, train_size= 0.8, random_state = 100)

    # part 7
    print len(training_data)
    print len(training_labels)

    # part 9
    classifier = KNeighborsClassifier(n_neighbors = 3)

    # part 10
    classifier.fit(training_data, training_labels)

    # part 11
    print classifier.score(validation_data, validation_labels)

    # part 12 and 15
    accuracies = []
    for k in range(1, 100):
        classifier = KNeighborsClassifier(n_neighbors = k)
        classifier.fit(training_data, training_labels)
        score = classifier.score(validation_data, validation_labels)
        print "For run {}: {}".format(k, score)
        accuracies.append(score)
    print "Max score: {}".format(max(accuracies))

    # part 14
    k_list = range(1, 100)

    # part 16 and 17
    plt.plot(k_list, accuracies)
    plt.title("Breast Cancer Classification Accuracy")
    plt.xlabel("k")
    plt.ylabel("Validation Accuracy")
    plt.show()


if __name__ == "__main__":
    main()
