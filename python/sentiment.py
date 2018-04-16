from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn import svm


def import_text():
    """Read in training file, removing capitalisation, and duplicates."""
    text_file = open('training.txt', 'r')
    text = set()
    for line in text_file:
        text.add(line.lower())
    return list(text)


def create_corpus(text):
    """Split raw text into sentiment and review."""
    corpus = []
    sentiment = []
    for line in text:
        sentiment.append(line.split('\t')[0])
        corpus.append(line.split('\t')[1])
    return corpus, sentiment


def extract_frequency_features(corpus):
    """Use tf-idf to extract features from the corpus"""
    frequency_vectorizer = TfidfVectorizer(analyzer='word', lowercase=False)
    frequency_features = frequency_vectorizer.fit_transform(corpus)
    return frequency_features.toarray()


def split_data(features, sentiment, training_size=0.90):
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
    """Evaluate classifier's f1-score."""
    return f1_score(actual_labels, predicted_labels, average='micro')


def process_data():
    """Import text file, convert to corpus, and extract features."""
    text = import_text()
    corpus, sentiment = create_corpus(text)
    features = extract_frequency_features(corpus)

    return features, sentiment, corpus


def run_classifier_split(classifier, train_data, test_data, train_labels,
                         test_labels):
    """Test a classifier using a 90:10 training/testing set."""
    train_classifier(classifier, train_data, train_labels)
    predicated_labels = test_classifier(classifier, test_data)
    return evaluate(test_labels, predicated_labels)


def run_classifier_fold(classifier, features, sentiment):
    """Test a classifier using a k-folded data set."""
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


def repeat_run_classifier_split(classifier, train_data, test_data, train_labels,
                                test_labels):
    """Test the average performance of a classifier."""
    accuracies = []
    for i in range(11):
        accuracies.append(
            run_classifier_split(classifier, train_data, test_data,
                                 train_labels, test_labels))
    return sum(accuracies) / len(accuracies)


def run_all_tests():
    """Test linear and rbf kernels with folded and split data."""
    classifiers = [svm.SVC(kernel='linear'),
                   svm.SVC(kernel='rbf'),
                   # svm.SVC(kernel='poly')
                   ]
    split_results = []
    fold_results = []
    features, sentiment, corpus = process_data()
    train_data, test_data, train_labels, test_labels = split_data(features,
                                                                  sentiment)

    for classifier in classifiers:
        split_result = repeat_run_classifier_split(classifier, train_data,
                                                   test_data, train_labels,
                                                   test_labels)
        fold_result = run_classifier_fold(classifier, features, sentiment)

        split_results.append(split_result)
        fold_results.append(fold_result)

    print 'kernel|data |result'
    for i, classifier in enumerate(classifiers):
        print '{0: <6}|split|{1}'.format(classifier.kernel, split_results[i])
        print '{0: <6}|fold |{1}'.format(classifier.kernel, fold_results[i])


def main():
    """System entry point."""
    run_all_tests()


if __name__ == '__main__':
    main()
