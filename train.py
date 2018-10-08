# from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import Dense, Embedding
# from keras.layers import LSTM
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import pickle

max_features = 2000

def create_model():
    embed_dim = 128
    lstm_out = 196
    model = Sequential()
    model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    return model

def save_model(model,path,weight):
    model_json = model.to_json()
    with open(path, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weight)

def load_model(path,weight):
    json_file = open(path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weight)
    loaded_model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    loaded_model._make_predict_function()
    return loaded_model

def train_model(model,X_train,Y_train):
    model.fit(X_train, Y_train, epochs = 7, batch_size=32, verbose = 1)

def open_set(path):
    return pd.read_json(path, encoding="utf-8")

def open_all_set(good,bad):
    good_set = open_set(good)
    bad_set = open_set(bad)
    return pd.concat([good_set,bad_set])

def get_words(all):
    words = all["text"].apply(lambda x: x.lower())
    words = words.apply(lambda x: re.sub('[^a-zA-zа-яА-Я0-9\s]','',x))
    words = words.apply(lambda x: re.sub('(\n|\xa0)',' ',x))
    return words

def create_tokenizer(words):
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(words.values)
    return tokenizer

def save_tokenizer(tokenizer,file):
    with open(file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer(file):
    with open(file, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def create_XY(tokenizer,all,words):
    X = tokenizer.texts_to_sequences(words.values)
    X = pad_sequences(X)
    Y = pd.get_dummies(all['note']).values
    return X,Y

def test_model(model, X_test, Y_test):
    validation_size = int(len(X_test)/2)

    X_validate = X_test[validation_size:]
    Y_validate = Y_test[validation_size:]
    X_test = X_test[:validation_size]
    Y_test = Y_test[:validation_size]
    score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = 32)
    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))
    pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
    for x in range(len(X_validate)):
        result = model.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
        if np.argmax(result) == np.argmax(Y_validate[x]):
            if np.argmax(Y_validate[x]) == 0:
                neg_correct += 1
            else:
                pos_correct += 1
        if np.argmax(Y_validate[x]) == 0:
            neg_cnt += 1
        else:
            pos_cnt += 1
    print("pos_acc", pos_correct/pos_cnt*100, "%")
    print("neg_acc", neg_correct/neg_cnt*100, "%")

def run():
    good = open_set("./dataset/good.json")
    bad = open_set("./dataset/bad.json")
    all = open_all_set("./dataset/good.json","./dataset/bad.json")
    words = get_words(all)


    # loading
    tokenizer = load_tokenizer('./model/tokenizer.pickle')

    # create_tokenizer(words)
    # save_tokenizer('./model/tokenizer.pickle')

    X,Y = create_XY(tokenizer,all,words)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)

    # model = create_model()
    # train_model(model,X_train,Y_train)
    # save_model(model,"model/model.json", "model/weights.h5")

    model = load_model("model/model.json", "model/weights.h5")
    test_model(model,X_test,Y_test)

if __name__ == '__main__':
    run()
