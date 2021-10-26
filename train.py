from pickle import load
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint
from pickle import dump
import tensorflow as tf
import sys
import zipfile
import os

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text
# load a pre-defined list of photo identifiers
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            descriptions[image_id].append(desc)
    return descriptions


# load photo features
def load_photo_features(filename, dataset):
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features


# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos,vocab_size):
    X1, X2, y = list(), list(), list()
    # walk through each image identifier
    for key, desc_list in descriptions.items():
        # walk through each description for the image
        for desc in desc_list:
            # encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0]
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # store
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return array(X1), array(X2), array(y)


# define the captioning model
def define_model(vocab_size, max_length,parameter1,dropout):
    # feature extractor model
    inputs1 = Input(shape=(4096, ))
    fe1 = Dropout(dropout)(inputs1)
    fe2 = Dense(parameter1, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, parameter1, mask_zero=True)(inputs2)
    se2 = Dropout(dropout)(se1)
    se3 = LSTM(parameter1)(se2)
    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(parameter1, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation="softmax")(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model

#Reading hyperparameter file
def read_hyperparameterFile(hyperparameterList,filepath):
	# list for keeping the best hyperparameters
	with open(filepath) as fp:
		line = fp.readline()
		hyperparameterList.append(line)
		while line:
			# print("Line {}: {}".format(cnt, line.strip()))
			line = fp.readline()
			hyperparameterList.append(line)
			#print(line)


# train dataset
def _main_():
    #load hyperparameters
    parameterList = []
    HPpath = sys.argv[2]
    hpfile =open(HPpath,"r")
    #read_hyperparameterFile(parameterList, HPpath)
    # load training dataset (6K)
    training = sys.argv[1].split("\\")
    trainPath = "."
    for i in range (1,len(training)-1):
        trainPath+="\\"+training[i]
    print(trainPath)
    with zipfile.ZipFile(sys.argv[1], "r") as zip_ref:
        zip_ref.extractall(trainPath)
    os.remove(sys.argv[1])
    filename = trainPath+"\\trainSet.txt"
    train = load_set(filename)
    print('Dataset: %d' % len(train))
    # descriptions
    train_descriptions = load_clean_descriptions('Data\\Temp\\descriptions.txt', train)
    print('Descriptions: train=%d' % len(train_descriptions))
    # photo features
    train_features = load_photo_features('Data\\Temp\\features.pkl', train)
    print('Photos: train=%d' % len(train_features))
    # prepare tokenizer
    tokenizer = create_tokenizer(train_descriptions)
    dump(tokenizer, open('Data\\Temp\\tokenizer.pkl', 'wb'))
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    # determine the maximum sequence length
    f = open("Data\\Temp\\max_length.txt", "w")
    max_l = max_length(train_descriptions)
    f.write(str(max_l))
    f.close()
    print('Description Length: %d' % max_l)
    # prepare sequences
    X1train, X2train, ytrain = create_sequences(tokenizer, max_l, train_descriptions, train_features,vocab_size)
    # define the model
    parameter = int(hpfile.readline())
    dropOut = float(hpfile.readline())
    hpfile.close()
    print("%%%%%%%%%%%%%%%%%",dropOut," ", parameter)
    model = define_model(vocab_size, max_l,parameter,dropOut)
    # fit model
    model.fit([X1train, X2train], ytrain, epochs=10, verbose=2)
    model.save("model.h5")
if __name__ == '__main__':
    _main_()
