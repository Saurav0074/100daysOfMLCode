import pandas as pd
import numpy as np
import itertools
from clean_data import  text_to_word_list
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from time import  time

import  seaborn as sns
import  datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, f1_score

from keras.preprocessing.sequence import pad_sequences
from  keras.models import  Model
from keras.layers import  Input, Embedding, LSTM, Merge
import  keras.backend as K
from keras.utils import plot_model
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
#nltk.download('stopwords')

mode = 'train'
stops = set(stopwords.words('english'))

################ Plot model loss and accuracy through epochs ###########
def plot_model_performance(train_loss, train_acc, train_val_loss, train_val_acc):
    """ Plot model loss and accuracy through epochs. """
 
    green = '#72C29B'
    orange = '#FFA577'
 
    with plt.xkcd():
        # plot model loss
        fig, ax1 = plt.subplots()
        ax1.plot(range(1, len(train_loss) + 1), train_loss, green, linewidth=5,
                 label='training')
        ax1.plot(range(1, len(train_val_loss) + 1), train_val_loss, orange,
                 linewidth=5, label='validation')
        ax1.set_xlabel('# epoch')
        ax1.set_ylabel('loss')
        ax1.tick_params('y')
        ax1.legend(loc='upper right', shadow=False)
        
        # plot model accuracy
        fig, ax2 = plt.subplots()
        ax2.plot(range(1, len(train_acc) + 1), train_acc, green, linewidth=5,
                 label='training')
        ax2.plot(range(1, len(train_val_acc) + 1), train_val_acc, orange,
                 linewidth=5, label='validation')
        ax2.set_xlabel('# epoch')
        ax2.set_ylabel('accuracy')
        ax2.tick_params('y')
        ax2.legend(loc='lower right', shadow=False)
    plt.show()

def remove_classes(df):
    df1 = df[df.label != 0]
    df2 = df[df.label != 1]

    df2 = df2[:100000]
    df = pd.concat([df1, df2], axis = 1)

    return df

train_path = "train_text.csv"
test_path = "test_text.csv"
word_embedding_path = '/home/saurav/Documents/GoogleNews-vectors-negative300.bin'
model_saving_dir = './model_weights'

train_data = pd.read_csv(train_path, sep='\t', names=['citance', 'reference', 'label'])
test_data = pd.read_csv(test_path, sep='\t', names=['citance', 'reference', 'label'])

print(len(train_data))
print(len(test_data))

train_data = train_data.sample(frac=1).reset_index(drop=True)
test_data = test_data.sample(frac=1).reset_index(drop=True)

print(train_data.groupby('label').count())
print(test_data.groupby('label').count())


vocab = dict()
inverse_vocab = ['<unk>']
word2vec = KeyedVectors.load_word2vec_format(word_embedding_path, binary=True)

sentence_cols = ['citance', 'reference']

for dataset in [train_data, test_data]:
    for index, row in dataset.iterrows():

        for q in sentence_cols:
            q2n = [] # question numbers
            for word in text_to_word_list(row[q]):
                if word in stops and word not in word2vec.vocab:
                    continue

                if word not in vocab:
                    vocab[word] = len(inverse_vocab)
                    q2n.append(len(inverse_vocab))
                    inverse_vocab.append(word)
                else:
                    q2n.append(vocab[word])

            dataset.set_value(index, q, q2n)


embedding_dim = 300
embeddings = 1 * np.random.randn(len(vocab) + 1, embedding_dim)
embeddings[0] = 0 # to ignore paddings

for word, index in vocab.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)

del word2vec

############# data set preparation ################

maxseqlen = max(train_data.citance.map(lambda x: len(x)).max(),
              train_data.reference.map(lambda x: len(x)).max(),
              test_data.citance.map(lambda x: len(x)).max(),
              test_data.reference.map(lambda x: len(x)).max())

validation_size = 10000
train_size = len(train_data) - validation_size

X = train_data[sentence_cols]
y = train_data['label']

X_test = test_data[sentence_cols]
y_test = test_data['label']

#print(X)
#print(y[0])

def create_model():
    hidden_num = 50
    gradient_clipping_norm = 1.25
    batch_size = 512
    n_epoch = 25

    def exponent_neg_manhattan_distance(left, right):
        return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims = True))


    left_input = Input(shape=(maxseqlen,), dtype='int32')
    right_input = Input(shape=(maxseqlen,), dtype='int32')

    embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], \
                                input_length = maxseqlen, trainable=False)

    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    shared_lstm = LSTM(hidden_num)

    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)

    malstm_distance = Merge(mode= lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                            output_shape = lambda x: (x[0][0], 1))([left_output, right_output])

    malstm = Model([left_input, right_input], [malstm_distance])

    # Adadelta optimizer, with gradient clipping by norm
    optimizer = Adadelta(clipnorm=gradient_clipping_norm)

    malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    return malstm

malstm = create_model()
print(malstm.summary())
plot_model(malstm, to_file="siamese.png", show_shapes=True)

if mode == 'train':
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size)

    X_train = {'left': X_train.citance, 'right': X_train.reference}
    X_val = {'left': X_val.citance, 'right': X_val.reference}
    
    y_train = y_train.values
    y_val = y_val.values

    for dataset, side in itertools.product([X_train, X_val], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], maxlen = maxseqlen)

    assert X_train['left'].shape == X_train['right'].shape
    assert  len(X_train['left']) == len(y_train)

    #################### Model Building ####################
    # assign class weights so that one instance of '1' = n instances of '0'
    class_weights = {0: 1., 1:380.}
    #class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train),
    #    y_train)

    # Start training
    n_epoch=200
    training_start_time = time()

    malstm_trained = malstm.fit([X_train['left'], X_train['right']], y_train, batch_size=400,
                                epochs=n_epoch, class_weight=class_weights,
                                validation_data=([X_val['left'], X_val['right']], y_val),
                                callbacks=[EarlyStopping(patience=8, verbose=1),
            ModelCheckpoint('./model_weights/class_weight_380.hdf5', verbose=1)])

    print("Training time finished.\n{} epochs in {}"\
        .format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))


    print(malstm_trained.history.keys())
    print(malstm_trained)
    plot_model_performance(
        train_loss=malstm_trained.history.get('loss', []),
        train_acc=malstm_trained.history.get('acc', []),
        train_val_loss=malstm_trained.history.get('val_loss', []),
        train_val_acc=malstm_trained.history.get('val_acc', [])
    )

else:
    X_test = {'left': X_test.citance, 'right': X_test.reference}
    y_test = y_test.values 
    for dataset, side in itertools.product([X_test], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], maxlen = maxseqlen)

    plot_model(malstm, to_file="model_arch.png", show_shapes=True)
    y_pred = malstm.predict([X_test['left'], X_test['right']])
    print("########### precision here #########",precision_recall_fscore_support(y_test, y_pred.round()))
