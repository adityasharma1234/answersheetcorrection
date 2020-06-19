from tensorflow.keras.models import Sequential, load_model, model_from_config
from tensorflow.keras.layers import Embedding, GRU,Dense, Dropout, Lambda, Flatten
#from tensorflow.keras.models import Sequential, load_model, model_from_config
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score

def get_model():

    model = Sequential()
    model.add(GRU(200, dropout=0.6, recurrent_dropout=0.6, input_shape=[1, 200], return_sequences=True))
    model.add(GRU(64, recurrent_dropout=0.6))
    model.add(Dropout(0.6))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()
    return model


cv = KFold(n_splits=5, shuffle=True)
results = []
predictionlist = []

count = 1
for traincv, testcv in cv.split(X):

    print("\n--------Fold {}--------\n".format(count))
    X_test, X_train, y_test, y_train = X.iloc[testcv], X.iloc[traincv], y.iloc[testcv], y.iloc[traincv]

    train = X_train['essay']
    test = X_test['essay']

    s = []
    for e in train:
        s +=tosentence(e, remove_stopwords = True)
    numberoffeaturess = 200
    minimumword = 40
    numberofworker = 4
    contet = 10
    downsampling = 1e-3
    model =Word2Vec(s, workers=numberofworker, size=numberoffeaturess, min_count = minimumword, window = contet, sample = downsampling)
    model.init_sims(replace=True)
    model.wv.save_word2vec_format("model1.bin", binary =True )
    c_train = []
    for es in train:
        c_train.append(toword(es, remove_stopwords=True))
    trainvector = AverageFeature(c_train, model, numberoffeaturess)
    c_test = []
    for es in test:
        c_test.append(toword( es, remove_stopwords=True ))
    testvector = AverageFeature( c_test, model, numberoffeaturess )
    trainvector = np.array(trainvector)
    testvector = np.array(testvector)
    trainvector = np.reshape(trainvector, (trainvector.shape[0], 1, trainvector.shape
    [1]))
    testvector = np.reshape(testvector, (testvector.shape[0], 1, testvector.shape[1]))

    model1 = get_model()
    model1.fit(trainvector, y_train, batch_size=64, epochs=15)
    yprediction = model1.predict(testvector)
    if count == 5:
         model1.save('model_gru.h5')
    yprediction = np.around(yprediction)
    result = cohen_kappa_score(y_test.values,yprediction,weights='quadratic')
    print("Kappa Score: {}".format(result))
    results.append(result)

    count += 1
