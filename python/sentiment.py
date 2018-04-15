from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
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
    train_data, test_data, train_labels, test_labels = train_test_split(
        features,
        sentiment,
        train_size=0.80,
        shuffle=True,
        random_state=1234)
    return train_data, test_data, train_labels, test_labels


def train_svm(train_data, train_labels):
    classifier_linear.fit(train_data, train_labels)


def test_svm(test_data):
    return classifier_linear.predict(test_data)


def evaluate(actual_results, predicted_results):
    return f1_score(actual_results, predicted_results, average='micro')


def main():
    text = import_text()
    corpus, sentiment = create_corpus(text)
    features = extract_features(corpus)
    train_data, test_data, train_labels, test_labels = split_data(features,
                                                                  sentiment)
    train_svm(train_data, train_labels)
    predicted_labels = test_svm(test_data)
    print evaluate(predicted_labels, test_labels)


if __name__ == '__main__':
    main()
