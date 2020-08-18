import json
import numpy as np
import os

from absl import app
from absl import logging
from stanfordcorenlp import StanfordCoreNLP
    
    
def ReadDataFile(filename):
    # load file
    sentences = list()
    len_sentences = list()
    with open(filename, 'r', encoding='utf-8') as textfile:
        for line in textfile:
            line = line.strip()
            tokens = line.split()
            sentences.append(line)
            len_sentences.append(len(tokens))
    return sentences, len_sentences


def FilterSentences(sentences, len_sentences, mean, std):
    # filter sentences 
    # take the sentences of length within range [mean - std, mean + std]
    filtered_sentences = list()
    for i in range(len(sentences)):
        if len_sentences[i] >= mean - std and len_sentences[i] <= mean + std:
            filtered_sentences.append(sentences[i])
    return filtered_sentences, len(filtered_sentences)


def DependencyParsing(sentences, num_sentences, out_pos_file_basename, out_dep_file_basename):
    # parsing with dependency parser
    file_index = 0
    current_min = 0
    sent_per_file = 10000
    current_max = min(sent_per_file, num_sentences - sent_per_file * file_index)
    
    sentences_with_pos = dict(zip(range(current_min, current_max), [[] for i in range(current_min, current_max)]))
    sentences_with_dep = dict(zip(range(current_min, current_max), [[] for i in range(current_min, current_max)]))
    
    stanfordnlp = StanfordCoreNLP('./stanford-corenlp-3.9.2') # initialize corenlp server in the background 
    properties = {'annotators': 'depparse', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
    
    logging.info('annotating {} set with stanford parser, total sentences to annotate: {}...'.format(out_pos_file_basename[14:-4], num_sentences))
    
    for i in range(len(sentences)):
        annotated = stanfordnlp.annotate(sentences[i], properties=properties)
        
        annotated = json.loads(annotated)
        
        dep_info = annotated['sentences'][0]['enhancedPlusPlusDependencies']
        pos_info = annotated['sentences'][0]['tokens']
        
        sentences_with_dep[i] = dep_info
        
        
        for token in pos_info:
            sentences_with_pos[i].append({'index':token['index'], 'word':token['word'], 'pos':token['pos']})
        
        if (i + 1) % sent_per_file == 0 or i == len(sentences) - 1: # output a file after every so many sentences 
            logging.info('\t{}/{}: {}% parsed'.format((i+1), num_sentences, round((i+1)/num_sentences*100, 2)))

            out_pos_file = '{}.{}.json'.format(out_pos_file_basename, file_index)
            out_dep_file = '{}.{}.json'.format(out_dep_file_basename, file_index)
            
            with open(out_pos_file, 'w', encoding='utf-8') as textfile:
                json.dump(sentences_with_pos, textfile)
            
            with open(out_dep_file, 'w', encoding='utf-8') as textfile:
                json.dump(sentences_with_dep, textfile)
            
            # reset filename, range for the next batch  
            file_index += 1
            current_min = sent_per_file * file_index
            current_max = min(sent_per_file, num_sentences - sent_per_file * file_index) + sent_per_file * file_index
            
            sentences_with_pos = dict(zip(range(current_min, current_max), [[] for i in range(current_min, current_max)]))
            sentences_with_dep = dict(zip(range(current_min, current_max), [[] for i in range(current_min, current_max)]))
            
    stanfordnlp.close()


def main(unused):

    # constants
    train_file_raw = 'samples50000_0206194500.train.txt'
    dev_file_raw = 'samples50000_0206194500.dev.txt'
    test_file_raw = 'samples50000_0206194500.test.txt'

    dataset_dir = './dataset_filtered'
    train_file = '{}/train.json'.format(dataset_dir)
    dev_file = '{}/dev.json'.format(dataset_dir)
    test_file = '{}/test.json'.format(dataset_dir)

    pos_file_dir = './dataset_pos'
    train_pos_file = '{}/train.pos'.format(pos_file_dir)
    dev_pos_file = '{}/dev.pos'.format(pos_file_dir)
    test_pos_file = '{}/test.pos'.format(pos_file_dir)

    dep_file_dir = './dataset_dep'
    train_dep_file = '{}/train.dep'.format(dep_file_dir)
    dev_dep_file = '{}/dev.dep'.format(dep_file_dir)
    test_dep_file = '{}/test.dep'.format(dep_file_dir)

    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)
        
    if not os.path.isdir(pos_file_dir):
        os.mkdir(pos_file_dir)

    if not os.path.isdir(dep_file_dir):
        os.mkdir(dep_file_dir)
    

    # std and avg of sentence length in total dataset
    sentences_tr, len_sentences_tr = ReadDataFile(train_file_raw)
    sentences_dv, len_sentences_dv = ReadDataFile(dev_file_raw)
    sentences_ts, len_sentences_ts = ReadDataFile(test_file_raw)

    len_sentences_tt = len_sentences_tr + len_sentences_dv + len_sentences_ts
    mean_total = np.mean(len_sentences_tt)
    std_total = np.std(len_sentences_tt)

    # process filtering dataset, keep sentences within proper length, output files for record
    sentences_tr, num_sentences_tr_flt = FilterSentences(sentences_tr, len_sentences_tr, mean_total, std_total)
    sentences_dv, num_sentences_dv_flt = FilterSentences(sentences_dv, len_sentences_dv, mean_total, std_total)
    sentences_ts, num_sentences_ts_flt = FilterSentences(sentences_ts, len_sentences_ts, mean_total, std_total)

    with open(train_file, 'w', encoding='utf-8') as textfile:
        json.dump(sentences_tr, textfile, indent=2)

    with open(dev_file, 'w', encoding='utf-8') as textfile:
        json.dump(sentences_dv, textfile, indent=2)

    with open(test_file, 'w', encoding='utf-8') as textfile:
        json.dump(sentences_ts, textfile, indent=2)

    # annotating  
    DependencyParsing(sentences_dv, num_sentences_dv_flt, dev_pos_file, dev_dep_file)
    DependencyParsing(sentences_ts, num_sentences_ts_flt, test_pos_file, test_dep_file)
    #DependencyParsing(sentences_tr, num_sentences_tr_flt, train_pos_file, train_dep_file)

if __name__=='__main__':
    app.run(main)