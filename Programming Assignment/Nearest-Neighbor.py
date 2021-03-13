from sklearn.datasets import fetch_openml
import numpy as np
import scipy as sp
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784')
data = mnist['data']
labels = mnist['target']

idx = np.random.RandomState(0).choice(70000, 11000)
train = data.loc[idx[:10000], :].astype(int)
train_labels = labels.loc[idx[:10000]]
test = data.loc[idx[10000:], :].astype(int)
test_labels = labels.loc[idx[10000:]]


def findNearestK(trainingImages, queryImage, k):
    distances = distance.cdist(np.array([queryImage]), trainingImages)
    return distances.reshape((distances.shape[1],)).argsort()[:k]


def KNN(trainingImages, labels, queryImage, k):
    nearestIndexes = findNearestK(trainingImages, queryImage, k)
    nearestLabels = np.asarray(labels[nearestIndexes], dtype=int)
    counts = np.bincount(nearestLabels)
    return np.argmax(counts)


def testAccuracy(train, train_labels, test, test_labels, n, k):
    count = 0
    for i in range(len(test)):
        guessedIndex = KNN(train[:n], train_labels[:n], test[i], k)
        if guessedIndex == int(test_labels[i]):
            count += 1
    return count / len(test)


def Q2():
    accuracy = testAccuracy(train.to_numpy(), train_labels.to_numpy(), test.to_numpy(), test_labels.to_numpy(), 1000,
                            10) * 100
    print(f"The accuracy is: {accuracy:}%")


def Q3_4():
    accuraciesK = np.zeros(100)
    accuraciesN = np.zeros(50)
    ns = range(100, 5100, 100)
    for i in range(50):
        accuraciesN[i] = testAccuracy(train.to_numpy(), train_labels.to_numpy(), test.to_numpy(),
                                      test_labels.to_numpy(),
                                      ns[i], 1)
    for k in range(0, 100):
        accuraciesK[k] = testAccuracy(train.to_numpy(), train_labels.to_numpy(), test.to_numpy(),
                                      test_labels.to_numpy(),
                                      1000, k + 1)

    print(f"best K is: {accuraciesK.argmax():}")

    fig = plt.figure()
    plot1 = fig.add_subplot(121)
    plot2 = fig.add_subplot(122)
    plot1.plot(range(1, 101), accuraciesK)
    plot2.plot(ns, accuraciesN)
    plot1.set_xlabel("K")
    plot1.set_ylabel("Probability of prediction")
    plot2.set_xlabel("N")
    plot2.set_ylabel("Probability of prediction")
    plot1.title.set_text("Accuracy of prediction by K")
    plot2.title.set_text("Accuracy of prediction by N")
    fig.savefig("temp.pdf")
    fig.show()


if __name__ == '__main__':
    Q2()
