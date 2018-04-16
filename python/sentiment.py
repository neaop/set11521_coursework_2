from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn import svm
from numpy import array

classifier_linear = svm.SVC(kernel='linear')
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


def split_data(training_size=90):
    """Split features into testing and training sets."""
    train_data, test_data, train_labels, test_labels = train_test_split(
        features,
        sentiment,
        train_size=training_size,
        shuffle=True)
    return train_data, test_data, train_labels, test_labels


def train_classifier(train_data, train_labels):
    """Train classifier with training data."""
    classifier_linear.fit(train_data, train_labels)


def test_classifier(test_data):
    """Test classifier with testing data."""
    return classifier_linear.predict(test_data)


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


def split_prediction():
    """Train, test, and evaluate classifier using split data set."""
    train_data, test_data, train_labels, test_labels = process_data()
    train_classifier(train_data, train_labels)
    predicted_labels = test_classifier(test_data)
    return evaluate(test_labels, predicted_labels)


def repeat_split_prediction():
    """Run the split prediction 10 times to calculate average f1-score."""
    accuracies = []
    for x in range(11):
        accuracies.append(split_prediction())
    return sum(accuracies) / len(accuracies)


def k_fold_prediction(folds=10):
    """Train, test and evaluate classifier using k-folded data set."""
    process_data()
    accuracies = []
    k_folder = KFold(n_splits=folds, shuffle=True)
    for train_index, test_index in k_folder.split(features):
        train_features = [features[i] for i in train_index]
        test_features = [features[i] for i in test_index]
        train_labels = [sentiment[i] for i in train_index]
        test_labels = [sentiment[i] for i in test_index]

        train_classifier(train_features, train_labels)
        predicted_labels = test_classifier(test_features)
        accuracies.append(evaluate(test_labels, predicted_labels))

    return sum(accuracies) / len(accuracies)


def main():
    """Entry point."""
    linear_svm_results = repeat_split_prediction()
    k_fold_results = k_fold_prediction()
    print 'split linear SVM :\t {0}'.format(linear_svm_results)
    print 'k-fold linear SVM :\t {0}'.format(k_fold_results)


if __name__ == '__main__':
    main()
