# BiDirLSTM

A simple bi-directional LSTM model for predicting target nouns given the context from the sentence. The input is the pre-trained Glove embedding sequence, the output is the predicted target word embedding (normalized). 


lstm_model.py - the bi-directional lstm model
process_wiki.ipnb - including data processing, making dataloader, training code samples
SampleNounDataLoader.ipnb - a customized dataloader which added an index elements into the input loader 
load_glove.py - load pre-trained glove vectors
parse_wiki.ipnb - split the sampled wikipedia dataset 
