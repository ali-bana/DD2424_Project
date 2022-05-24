import torch
from mnist_svhn import get_data_loader
import numpy as np
from sklearn.cluster import KMeans
from test_utils import get_accuracy, get_latent_loader
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


def k_means_clustering_test(model, loader, mode):
    mnist, svhn, labels = get_latent_loader(model, loader, mode, 1000)
    # digit clustering with mnist
    kmeans = KMeans(10, max_iter=1000)
    label_mnist = kmeans.fit_predict(mnist, None)
    mnist_accuracy = get_accuracy(labels, label_mnist)

    # digit clustering using svhn
    kmeans = KMeans(10, max_iter=1000)
    label_svhn = kmeans.fit_predict(svhn, None)
    svhn_accuracy = get_accuracy(labels, label_svhn)
    # digit clustering with both
    kmeans = KMeans(10, max_iter=1000)
    label_both = kmeans.fit_predict(np.concatenate([mnist, svhn]), None)
    both_accuracy = get_accuracy(np.concatenate([labels, labels]), label_both)

    # modal clustering
    kmeans = KMeans(2, max_iter=1000)
    label_mode = kmeans.fit_predict(np.concatenate([mnist, svhn]), None)
    mode_accuracy = get_accuracy(np.concatenate(
        [np.ones(labels.shape), np.zeros(labels.shape)]), label_mode)

    return mnist_accuracy, svhn_accuracy, both_accuracy, mode_accuracy


def classifier_test(model, train_loader, test_loader, mode):
    mnist_train, svhn_train, labels_train = get_latent_loader(
        model, train_loader, mode, 3000)
    mnist_test, svhn_test, labels_test = get_latent_loader(
        model, test_loader, mode, 500)

    mnist_classifier = Perceptron()
    mnist_classifier.fit(mnist_train, labels_train)
    mnist_predict = mnist_classifier.predict(mnist_test)
    accuracy_mnist = accuracy_score(labels_test, mnist_predict)

    svhn_classifier = Perceptron()
    svhn_classifier.fit(svhn_train, labels_train)
    svhn_predict = svhn_classifier.predict(svhn_test)
    accuracy_svhn = accuracy_score(labels_test, svhn_predict)

    both_classifier = Perceptron()
    both_classifier.fit(np.concatenate(
        [mnist_train, svhn_train]), np.concatenate([labels_train, labels_train]))
    both_predict = svhn_classifier.predict(
        np.concatenate([mnist_test, svhn_test]))
    accuracy_both = accuracy_score(np.concatenate(
        [labels_test, labels_test]), both_predict)

    mode_classifier = Perceptron()
    mode_classifier.fit(np.concatenate(
        [mnist_train, svhn_train]), np.concatenate([np.ones(labels_train.shape), np.zeros(labels_train.shape)]))
    mode_predict = mode_classifier.predict(
        np.concatenate([mnist_test, svhn_test]))
    accuracy_mode = accuracy_score(np.concatenate(
        [np.ones(labels_test.shape), np.zeros(labels_test.shape)]), mode_predict)

    return accuracy_mnist, accuracy_svhn, accuracy_both, accuracy_mode


if __name__ == '__main__':
    train_loader, test_loader, val_loader = get_data_loader(25, 'data/')

    model = torch.load('saves/model_trying', map_location=torch.device('cpu'))
    # print(k_means_clustering_test(model, val_loader, 'mmvae'))
    # print(classifier_test(model, train_loader, test_loader, 'mmvae'))

    # model = torch.load('saves/final_pvae', map_location=torch.device('cpu'))
    # print(k_means_clustering_test(model, val_loader, 'pvae'))
    # print(classifier_test(model, train_loader, test_loader, 'pvae'))
