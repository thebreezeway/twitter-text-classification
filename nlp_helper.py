import numpy as np
import preprocessor as p

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def read_glove_vecs(glove_file):
    with open(glove_file, 'r',encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

    
def load_embeddings_dict(path):
    try:
        return  np.load(path+'/words.npy')[()],\
                np.load(path+'/embedding_vec.npy')[()]

    except FileNotFoundError:
        print("未在以下路径找到相应embedding数据:\n" + \
                path + '/words.npy' + \
                path + '/embedding_vec.npy'
        )

def load_words_and_embeddings(path):
    
    try:
        return  np.load(path+'/words.npy')[()],\
                np.load(path+'/embedding_vec.npy')[()]

    except FileNotFoundError:
        print("未在以下路径找到相应embedding数据:\n" + \
                path + '/words.npy' + \
                path + '/embedding_vec.npy'
        )
    
def load_char_embeddings(path):
    '''
    Returns:
    char_to_index -- dict1
    ndex_to_char -- dict2
    index_to_vec -- dict3
    '''
    try:
        return np.load(path+'/char_to_index.npy')[()], \
            np.load(path+'/index_to_char.npy')[()], \
            np.load(path+'/index_to_vec_char.npy')[()]
    except FileNotFoundError:
        print("未在以下路径找到相应embedding数据:\n" + \
                path + '/char_to_index.npy\n' +\
                path + '/index_to_char.npy\n' +\
                path + '/index_to_vec_char.npy'
        )
def load_corpus(path):

    with open(path, 'r') as f:
        data = f.readlines()

    return data

def replace_hashtags(tweet):
    
    p.set_options(p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.URL, p.OPT.RESERVED)
    t=p.parse(tweet)
    if t.hashtags:
        for i in t.hashtags:
            tweet = tweet[:i.start_index] + ' ' + tweet[i.start_index+1:]

    return tweet

def clean_url_replace_hashtag(tweet):
    return p.clean(replace_hashtags(tweet))
