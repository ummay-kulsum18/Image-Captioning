import train as sm
from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
import sys
import zipfile
import pickle
import os
import shutil
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model

def max_length(descriptions):
    lines = sm.to_lines(descriptions)
    return max(len(d.split()) for d in lines)
# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo ,sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    # calculate BLEU score
    result = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
    print('BLEU-Pure: %f' % corpus_bleu(actual, predicted))
    #print('BLEU-Pure_sentence: %f' % sentence_bleu(actual, predicted,weights=(1.0,0,0,0)))
    #print('BLEU-Pure_sentence: %f' % sentence_bleu(actual, predicted))
    return result

def extract_features(filename):
    # load the model
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # load the photo
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature

# prepare tokenizer on train set
def _main_():
    # load training dataset (6K)

    testing = sys.argv[1].split("\\")
    testPath = "."
    for i in range(1, len(testing) - 1):
        testPath += "\\" + testing[i]
    print(testPath)
    with zipfile.ZipFile(sys.argv[1], "r") as zip_ref:
        zip_ref.extractall(testPath)
    os.remove(sys.argv[1])
    filename = testPath+'\\testSet.txt'
    test = sm.load_set(filename)
    print('Dataset: %d' % len(test))
    # descriptions
    test_descriptions = sm.load_clean_descriptions('Data\\Temp\\descriptions.txt', test)
    print('Descriptions: test=%d' % len(test_descriptions))
    # photo features
    test_features = sm.load_photo_features('Data\\Temp\\features.pkl', test)
    print('Photos: test=%d' % len(test_features))

    f = open("Data\\Temp\\max_length.txt")
    length_max = int(f.readline())
    f.close()
    with (open("Data\\Temp\\tokenizer.pkl", "rb")) as openfile:
        tokenizer=pickle.load(openfile)


    # load the model
    filename = sys.argv[2]
    model = load_model(filename)
    # evaluate model
    r = evaluate_model(model, test_descriptions, test_features, tokenizer, length_max)
    print("test result:",r)
    shutil.rmtree("Data\\Temp")
    """
    f = open("Data\\Temp\\max_length.txt")
    length_max = int(f.readline())
    f.close()
    with (open("Data\\Temp\\tokenizer.pkl", "rb")) as openfile:
        tokenizer = pickle.load(openfile)
    model = load_model("model.h5")
    photo = extract_features('example.jpg')
    # generate description
    description = generate_desc(model, tokenizer, photo, length_max)
    print(description)
    """
if __name__ == '__main__':
    _main_()
