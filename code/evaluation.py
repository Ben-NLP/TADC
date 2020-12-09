from keras.utils.np_utils import to_categorical
import numpy as np

def cal_match(pred, gold):
    length = len(pred)
    match = 0
    i = 0
    gold_num = 0

    while i < length:
        if gold[i] == 1:
            gold_num += 1
            begin = i
            i += 1
            while i < length:
                if gold[i] == 0 or gold[i] == 1:
                    break
                i += 1
            
            if (gold[begin:i] == pred[begin:i]).all():
                if i == length:
                    match += 1
                elif pred[i] != 2:
                    match += 1
        else:
            i += 1

    return [gold_num, match]

def cal_sum(y, test_seq):
    temp = (y == 1) * (test_seq > 0)
    res = np.sum(temp)
    return res

def my_evaluate(pred, gold, test_seq):
    pred_sum = cal_sum(pred, test_seq)
    gold_sum = cal_sum(gold, test_seq)

    total_match = 0
    count = 0
    for i in range(len(pred)):
        gold_num, match = cal_match(pred[i], gold[i])
        total_match += match
        if match < gold_num:
            count += 1

    pre = total_match*1. / pred_sum
    rec = total_match*1. / gold_sum
    f1 = 2*pre*rec / (pre+rec)

    return pre, rec, f1

def evaluate(model, test_bow, test_seq, test_y, batch_size):
    pred_y = np.zeros((test_seq.shape[0], test_seq.shape[1]), np.int32)

    for offset in range(0, test_seq.shape[0], batch_size):
        batch_bow = test_bow[offset:offset+batch_size]
        batch_seq = test_seq[offset:offset+batch_size]
        batch_y = test_y[offset:offset+batch_size]
        _, _, _, batch_pred_y = model.predict_on_batch([batch_bow, batch_seq])
        batch_pred_y = np.argmax(batch_pred_y, axis=2)
        pred_y[offset:offset+batch_size,:batch_pred_y.shape[1]] = batch_pred_y

    assert pred_y.shape == test_seq.shape
    assert pred_y.shape == test_y.shape
    
    return my_evaluate(pred_y, test_y, test_seq)