import train as t
import sys
import zipfile
import os
import test as ts
from keras.models import load_model
from pickle import dump
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def tune():
    tuning =sys.argv[1].split("\\")
    validation= sys.argv[2].split("\\")
    tunePath = "."
    vPath = "."
    for i in range (1,len(tuning)-1):
        tunePath+="\\"+tuning[i]
    print(tunePath)

    for i in range (1,len(validation)-1):
        vPath+="\\"+validation[i]
    print(vPath)

    with zipfile.ZipFile(sys.argv[1], "r") as zip_ref:
        zip_ref.extractall(tunePath)
    os.remove(sys.argv[1])
    with zipfile.ZipFile(sys.argv[2], "r") as zip_ref:
        zip_ref.extractall(vPath)
    os.remove(sys.argv[2])


    parameter1 =[128,256,512]
    dropout=[0.25,0.5,0.65]

    filename = tunePath+'\\trainSet.txt'
    train = t.load_set(filename)
    print('Dataset: %d' % len(train))
    # descriptions
    train_descriptions = t.load_clean_descriptions('Data\\Temp\\descriptions.txt', train)
    print('Descriptions: train=%d' % len(train_descriptions))
    # photo features
    train_features = t.load_photo_features('Data\\Temp\\features.pkl', train)
    print('Photos: train=%d' % len(train_features))
    # prepare tokenizer
    tokenizer = t.create_tokenizer(train_descriptions)
    dump(tokenizer, open('Data\\Temp\\tokenizer.pkl', 'wb'))
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    # determine the maximum sequence length
    max_length = t.max_length(train_descriptions)
    X1train, X2train, ytrain = t.create_sequences(tokenizer, max_length, train_descriptions, train_features, vocab_size)

    filename =vPath+'\\validationSet.txt'
    test = t.load_set(filename)
    print('Dataset: %d' % len(test))
    # descriptions
    test_descriptions = t.load_clean_descriptions('Data\\Temp\\descriptions.txt', test)
    print('Descriptions: test=%d' % len(test_descriptions))
    # photo features
    test_features = t.load_photo_features('Data\\Temp\\features.pkl', test)
    print('Photos: test=%d' % len(test_features))
    file=open('tuning_results.txt','w')
    '''
    for i in parameter1:
        name="model"+str(i)
        model=t.define_model(vocab_size,max_length,i,.5)
        model.fit([X1train, X2train], ytrain, epochs=10, verbose=2)
        model.save("Data\\Temp\\"+name+".h5")
        r=ts.evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
        file.write(str(i)+ " "+ str(r)+ "\n")
        #file.write(" ")
        #file.write(r)
        #file.write("\n")
    for j in dropout:
        name = "model" + str(j)
        model=t.define_model(vocab_size, max_length, 256, j)
        model.fit([X1train, X2train], ytrain, epochs=10, verbose=2)
        model.save("Data\\Temp\\"+name+".h5")
        r=ts.evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
        file.write(str(j)+" "+str(r)+"\n")
        #file.write(" ")
        #file.write(r)
        #file.write("\n")
    file.close()
    '''
    max_BLEU = -9
    max_parameter_1=-1
    max_dropout =-1
    for i in parameter1:
        for j in dropout:
            name = "model" + str(j)+"_"+str(i)
            model = t.define_model(vocab_size, max_length, i, j)
            model.fit([X1train, X2train], ytrain, epochs=10, verbose=2)
            model.save("Data\\Temp\\" + name + ".h5")
            r = ts.evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
            if(r>max_BLEU):
                max_BLEU=r
                max_parameter_1=i
                max_dropout=j
            file.write(str(j) + " "+str(i)+" "+ str(r) + "\n")
    file.close()
    file1=open('hyperparameter.txt','w')
    file1.write(str(max_parameter_1)+"\n"+str(max_dropout))




tune()