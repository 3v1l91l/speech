from keras.models import load_model
from lib import get_path_label_df, prepare_data, get_specgrams
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from sklearn import preprocessing
import os

lb = preprocessing.LabelBinarizer()
train_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']
trained_labels = train_words + ['unknown']
all_folders = next(os.walk('../input/train/audio/'))
all_labels = [x for x in all_folders if x != '_background_noise']
cor_pred = []
all_num = []
labelbinarizer = LabelBinarizer()
labelbinarizer.fit(trained_labels)
for testing_label in train_words:
    actual_label = testing_label
    if testing_label not in train_words:
        actual_label = 'unknown'
    test = prepare_data(get_path_label_df('../input/train/audio/'+ testing_label + '/', '*.wav'))
    paths = test.path.tolist()
    predictions = []
    shape = (99, 161, 1)

    model = load_model('model.model')
    for path in paths:
        specgram = get_specgrams([path], shape)
        pred = model.predict(np.array(specgram))
        predictions.extend(pred)

    precicted_labels = [labelbinarizer.inverse_transform(p.reshape(1, -1))[0] for p in predictions]
    print('Predicted label: {}'.format(testing_label))
    print('Len labels: {}'.format(len(precicted_labels)))
    cor = np.sum(np.array(precicted_labels) == actual_label)
    cor_pred.append(cor)
    all_num.append(len(np.array(precicted_labels)))
    print('Pct detected: {}'.format((100 * cor) / len(precicted_labels)))

print('Overall correctly predicted: {}%'.format(100 * np.sum(cor_pred) / np.sum(all_num)))


# test['labels'] = labels
# test.path = test.path.apply(lambda x: str(x).split('/')[-1])
# submission = pd.DataFrame({'fname': test.path.tolist(), 'label': labels})
# submission.to_csv('submission.csv', index=False)
