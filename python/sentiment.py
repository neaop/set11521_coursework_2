from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn import svm
from numpy import array

corpus = []
sentiment = []
features = array([])


def import_text():
    """Read in training file, removing duplicates."""
    data_file = open('training.txt', 'r')
    data = []
    for line in data_file:
        data.append(line)
    return set(data)


def create_corpus(text):
    """Split raw text into sentiment and review."""
    if not corpus and not sentiment:
        for line in text:
            sentiment.append(line.split('\t')[0])
            corpus.append(line.split('\t')[1])
        return corpus, sentiment


def extract_tfidf_features():
    """Utilise tf-idf to extract features from the corpus"""
    global features
    if not features.any():
        tfidf_vectorizer = TfidfVectorizer(analyzer='word', lowercase=True)
        tfidf_features = tfidf_vectorizer.fit_transform(corpus)
        features = tfidf_features.toarray()


def split_data(training_size=0.90):
    """Split features into testing and training sets."""
    train_data, test_data, train_labels, test_labels = train_test_split(
        features,
        sentiment,
        train_size=training_size,
        shuffle=True)
    return train_data, test_data, train_labels, test_labels


def train_classifier(classifier, train_data, train_labels):
    """Train a classifier with testing data."""
    classifier.fit(train_data, train_labels, sample_weight=None)


def test_classifier(classifier, test_data):
    """Test a classifier with testing data."""
    predication = classifier.predict(test_data)
    return predication


def evaluate(actual_labels, predicted_labels):
    """Evaluate classifiers f1-score."""
    return f1_score(actual_labels, predicted_labels, average='micro')


def process_data():
    """Import text file, convert to corpus, and extract features."""
    text = import_text()
    create_corpus(text)
    extract_tfidf_features()
    train_data, test_data, train_labels, test_labels = split_data()
    return train_data, test_data, train_labels, test_labels


def k_fold_prediction(folds=10):
    """Train, test and evaluate classifier using k-folded data set."""
    process_data()
    linear_accuracies = []
    rbf_accuracies = []
    k_folder = KFold(n_splits=folds, shuffle=True)

    for train_index, test_index in k_folder.split(features):
        train_data = [features[i] for i in train_index]
        test_data = [features[i] for i in test_index]
        train_labels = [sentiment[i] for i in train_index]
        test_labels = [sentiment[i] for i in test_index]

        train_classifiers(train_data, train_labels)
        linear_predicted, rbf_predicted = test_classifiers(test_data)
        linear_accuracies.append(evaluate(test_labels, linear_predicted))
        rbf_accuracies.append(evaluate(test_labels, rbf_predicted))

    average_linear = sum(linear_accuracies) / len(linear_accuracies)
    average_rbf = sum(rbf_accuracies) / len(rbf_accuracies)

    return average_linear, average_rbf


def run_classifier_fold(classifier, featuers):
    k_folder = KFold(n_splits=10, shuffle=True)
    accuracies = []
    for train_index, test_index in k_folder.split(features):
        train_data = [features[i] for i in train_index]
        test_data = [features[i] for i in test_index]
        train_labels = [sentiment[i] for i in train_index]
        test_labels = [sentiment[i] for i in test_index]

        train_classifier(classifier, train_data, train_labels)
        predicated_labels = test_classifier(classifier, test_data)
        accuracies.append(evaluate(test_labels, predicated_labels))
    return sum(accuracies) / len(accuracies)


def run_classifier_split(classifier, train_data, test_data, train_labels,
                         test_labels):
    train_classifier(classifier, train_data, train_labels)
    predicated_labels = test_classifier(classifier, test_data)
    return evaluate(test_labels, predicated_labels)


def repeat_run_classifier_split(classifier, train_data, test_data, train_labels,
                                test_labels):
    accuracies = []
    for i in range(11):
        accuracies.append(
            run_classifier_split(classifier, train_data, test_data,
                                 train_labels, test_labels))
    return sum(accuracies) / len(accuracies)


def main():
    """System entry point."""
    classifiers = [svm.SVC(kernel='linear'), svm.SVC(kernel='rbf')]
    split_results = []
    fold_results = []

    train_data, test_data, train_labels, test_labels = process_data()

    for classifier in classifiers:
        split_result = repeat_run_classifier_split(classifier, train_data,
                                                   test_data, train_labels,
                                                   test_labels)
        fold_result = run_classifier_fold(classifier, features)

        split_results.append(split_result)
        fold_results.append(fold_result)

    for i, classifier in enumerate(classifiers):
        print 'split {0: <6} : {1}'.format(classifier.kernel, split_results[i])
        print 'fold  {0: <6} : {1}'.format(classifier.kernel, fold_results[i])

    # linear_split_results, rbf_split_results = repeat_split_prediction()
    # linear_fold_results, rbf_fold_results = k_fold_prediction()
    # print 'data | kernel | result'
    # print 'split| linear | {0}'.format(linear_split_results)
    # print 'split| rbf    | {0}'.format(rbf_split_results)
    # print 'fold | linear | {0}'.format(linear_fold_results)
    # print 'fold | rbf    | {0}'.format(rbf_fold_results)


if __name__ == '__main__':
    main()
