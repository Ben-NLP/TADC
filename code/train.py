from keras.backend import set_session
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
import tensorflow as tf

from layers import create_model
from evaluation import evaluate
from config import args

import numpy as np
import json
import time
import random
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def get_batches_num(source, batch_size):
    samples_num = len(source)
    batches_num = int(samples_num / batch_size)
    if samples_num % batch_size != 0:
        batches_num += 1
    return batches_num

def print_top_words(phi, vocab_bow, top_k):
    print('top %d words:' % top_k)
    for k, phi_k in enumerate(phi):
        topic_words = [vocab_bow[idx] for idx in np.argsort(phi_k)[:-top_k-1:-1]]
        print(' '.join(topic_words))  

def train(model, model_dir, epochs, train_batches_num, batch_size, train_bow, valid_bow, train_seq, valid_seq, train_y, valid_y):
    min_loss = np.inf
    for epoch in range(epochs):
        for index in range(train_batches_num):
            batch_bow = train_bow[index*batch_size: (index+1)*batch_size]
            batch_seq = train_seq[index*batch_size: (index+1)*batch_size]
            batch_y = train_y[index*batch_size: (index+1)*batch_size]

            temp = np.zeros((len(batch_bow),))
            loss, kl, nnl, reg, cat = model.train_on_batch([batch_bow, batch_seq], [temp, temp, temp, batch_y])

        temp = np.zeros((len(valid_bow),))
        loss, kl, nnl, reg, cat = model.test_on_batch([valid_bow, valid_seq], [temp, temp, temp, valid_y])
        print('epoch%d; valid_loss: %.5lf | kl+nnl: %.5lf kl: %.5lf nnl: %.5lf reg: %.5lf | cat: %.5lf' % (epoch, loss, kl+nnl, kl, nnl, reg, cat))

        if cat < min_loss:
            min_loss = cat
            model.save(model_dir)

        shuffle_idx = np.random.permutation(len(train_bow))
        train_bow = train_bow[shuffle_idx]
        train_seq = train_seq[shuffle_idx]
        train_y = train_y[shuffle_idx]

def execute(data_dir, model_dir, hidden_size, topic_num, shortcut, top_k, dropout, lr, epochs, num_valid, batch_size):
    vocab_bow = json.load(open(data_dir+'vocab_bow.json'))
    vocab_bow = sorted(vocab_bow.items(), key=lambda x:x[1])
    vocab_bow = [item[0] for item in vocab_bow]

    train_bow = np.load(data_dir+'train_bow.npy')
    valid_bow = train_bow[-num_valid:]
    train_bow = train_bow[:-num_valid]
    test_bow = np.load(data_dir+'test_bow.npy')

    train_seq = np.load(data_dir+'train_seq.npy')
    test_seq = np.load(data_dir+'test_seq.npy')
    valid_seq = train_seq[-num_valid:]
    train_seq = train_seq[:-num_valid]

    train_y = np.load(data_dir+'train_y.npy')
    test_y = np.load(data_dir+'test_y.npy')
    valid_y = train_y[-num_valid:]
    valid_y = to_categorical(valid_y, num_classes=3)
    train_y = train_y[:-num_valid]
    train_y = to_categorical(train_y, num_classes=3)

    topic_emb = np.load(data_dir+'topic_emb.npy')
    bow_emb = np.load(data_dir+'bow_emb.npy')
    seq_emb = np.load(data_dir+'seq_emb.npy')

    train_batches_num = get_batches_num(train_bow, batch_size)

    model = create_model(train_bow.shape[1], hidden_size, topic_num, shortcut, train_seq.shape[1], dropout, lr, topic_emb, bow_emb, seq_emb)
    plot_model(model, to_file='model.png', show_shapes=True)
    train(model, model_dir, epochs, train_batches_num, batch_size, train_bow, valid_bow, train_seq, valid_seq, train_y, valid_y)

    model = create_model(test_bow.shape[1], hidden_size, topic_num, shortcut, test_seq.shape[1], dropout, lr, topic_emb, bow_emb, seq_emb)
    model.load_weights(model_dir, by_name=True)
    pre, rec, f1 = evaluate(model, test_bow, test_seq, test_y, batch_size)
    print('pre: %.5lf rec: %.5lf f1: %.5lf' % (pre, rec, f1))
    print()
        
    tv, wv = model.get_layer('d1').get_weights()
    tw = np.dot(tv, np.transpose(wv))
    print_top_words(tw, vocab_bow, top_k)

def main():
    domain = sys.argv[1]
    data_dir = args.data_dir + domain + '/'
    model_dir = args.model_dir + domain

    execute(data_dir, model_dir, args.hidden_size, args.topic_num, args.shortcut, args.top_k, 
            args.dropout, args.lr, args.epochs, args.num_valid, args.batch_size)

if __name__ == '__main__':
    main()