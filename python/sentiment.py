from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

classifier_linear = svm.SVC(kernel='linear')


def import_text():
    """Read in training file, removing duplicates, and capitalisation."""
    data_file = open('training.txt', 'r')
    data = []
    for line in data_file:
        data.append(line.lower())
    return set(data)


def create_corpus(text):
    sentiment = []
    corpus = []
    for line in text:
        sentiment.append(line.split('\t')[0])
        corpus.append(line.split('\t')[1])
    return corpus, sentiment


def extract_features(corpus):
    vectorizer = CountVectorizer(
        analyzer='word',
        lowercase=False,
    )
    features = vectorizer.fit_transform(corpus)
    return features.toarray()


def split_data(features, sentiment):
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        sentiment,
        train_size=0.80,
        shuffle=True,
        random_state=1234)
    return x_train, x_test, y_train, y_test


def train_svm(x_train, y_train):
    classifier_linear.fit(x_train, y_train)


def test_svm(x_test):
    return classifier_linear.predict(x_test)


def evaluate(actual_results, test_results):
    print(accuracy_score(actual_results, test_results))


def main():
    text = import_text()
    corpus, sentiment = create_corpus(text)
    features = extract_features(corpus)
    x_train, x_test, y_train, y_test = split_data(features, sentiment)
    train_svm(x_train, y_train)
    predicted = test_svm(x_test)
    evaluate(y_test, predicted)


if __name__ == '__main__':
    main()
