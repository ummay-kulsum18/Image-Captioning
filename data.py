import zipfile
import os
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import random
import sys
import string

# extract features from each photo in the directory
def extract_features(directory,f):
    # load the model
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # summarize
    print(model.summary())
    # extract features from each photo
    features = dict()
    for name in os.listdir(directory):
        # load an image from file
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature
        #print('>%s' % name)
        f.write(name)
        f.write('\n')
    f.close()
        #print(feature.shape)
    return features

def get_trainingFile():
    line = open(dir_path+"\\Temp\\images.txt").readlines()
    random.shuffle(line)
    open(dir_path+"\\Temp\\images.txt", 'w').writelines(line)

    with open(dir_path+"\\Temp\\images.txt") as f:
        content = f.readlines()
    f = open("TrainSet.txt", "w")
    f1= open("ValidationSet.txt", "w")
    f2 = open("TestSet.txt", "w")
    f3 = open("TrainSet10.txt", "w")
    f4 = open("TrainSet90.txt", "w")
    f5 = open("ValidationSet3.txt", "w")
    for i in range(0, 120):

        name = content[i]
        if (i < 10):
            f3.write(name)
        if (i>=50 and i<100 ):
            f4.write(name)
        f.write(name)
    f3.close()
    f4.close()
    f.close()

    for i in range(120, 135):
        name = content[i]
        if(i<123):
            f5.write(name)
        f1.write(name)
    f1.close()
    f5.close()

    for i in range(135, 150):
        name = content[i]
        f2.write(name)
    f2.close()

    zip = zipfile.ZipFile(dir_path+"\\Train\\Best_hyperparameter_80_percent\\data.zip","w")
    file = zip.write("TrainSet.txt")
    os.remove("TrainSet.txt")

    zip = zipfile.ZipFile(dir_path + "\Train\\Under_10_min_training\\data.zip", "w")
    os.rename("TrainSet10.txt","TrainSet.txt")
    file = zip.write("TrainSet.txt")
    os.remove("TrainSet.txt")

    zip = zipfile.ZipFile(dir_path + "\\Train\\Under_90_min_tuning\\data.zip", "w")
    os.rename("TrainSet90.txt", "TrainSet.txt")
    file = zip.write("TrainSet.txt")
    os.remove("TrainSet.txt")

    zip = zipfile.ZipFile(dir_path + "\\Validation\\Validation_10_percent\\data.zip", "w")
    file = zip.write("ValidationSet.txt")
    os.remove("ValidationSet.txt")

    zip = zipfile.ZipFile(dir_path + "\\Validation\\3_samples\\data.zip", "w")
    os.rename("validationSet3.txt", "ValidationSet.txt")
    file = zip.write("ValidationSet.txt")
    os.remove("ValidationSet.txt")

    zip = zipfile.ZipFile(dir_path + "\\Test\\Test_10_percent\\\data.zip", "w")
    file = zip.write("TestSet.txt")
    os.remove("TestSet.txt")



# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# extract descriptions for images
def load_descriptions(doc):
    mapping = dict()
    # process lines
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # remove filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        # create the list if needed
        if image_id not in mapping:
            mapping[image_id] = list()
        # store description
        mapping[image_id].append(image_desc)
    return mapping


def clean_descriptions(descriptions):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word) > 1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] = ' '.join(desc)


# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
    # build a list of all description strings
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc


# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()




dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path + "\\Data"
with zipfile.ZipFile(sys.argv[1], "r") as zip_ref:
    zip_ref.extractall(dir_path)
os.remove(sys.argv[1])

os.makedirs(dir_path + "\\Train\\Under_10_min_training", exist_ok=True)
os.makedirs(dir_path + "\\Train\\Under_90_min_tuning", exist_ok=True)
os.makedirs(dir_path + "\\Train\\Best_hyperparameter_80_percent", exist_ok=True)

os.makedirs(dir_path + "\\Test", exist_ok=True)
os.makedirs(dir_path + "\\Validation\\3_samples", exist_ok=True)
os.makedirs(dir_path + "\\Validation\\Validation_10_percent", exist_ok=True)
os.makedirs(dir_path + "\\Test\\Test_10_percent", exist_ok=True)

os.makedirs(dir_path + "\\Temp", exist_ok=True)


directory = dir_path+'\\data_150\\Flicker8k_Dataset'
# opening a file to save the names of images
f = open(dir_path+"\\Temp\\images.txt", "w")
features = extract_features(directory, f)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open(dir_path+'\\Temp\\features.pkl', 'wb'))
get_trainingFile()


filename = dir_path+'\\data_150\\Output.txt'
# load descriptions
doc = load_doc(filename)
# parse descriptions
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))
# clean descriptions
clean_descriptions(descriptions)
# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
#fp = open(dir_path+"\\Temp\\vocab_size.txt","w")
#fp.write(str(len(vocabulary)))
print('Vocabulary Size: %d' % len(vocabulary))

# save to file
save_descriptions(descriptions,dir_path+'\\Temp\\descriptions.txt')
