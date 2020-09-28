import pickle

infile = open('meme/train_feature.json','rb')


features = pickle.load(infile)

print(features[0])

infile.close()