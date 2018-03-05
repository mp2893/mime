import cPickle as pickle
from sklearn.model_selection import train_test_split
import numpy as np
import sys, os, pdb
import tensorflow as tf


def load_data(input_path, min_threshold=5, max_threshold=150, seed=1234, train_ratio=1., association_threshold=0.2, seqs_name='.seqs', labels_name='.labels', filter_codes=True):
    seqs = pickle.load(open(input_path + seqs_name, 'rb'))
    labels = pickle.load(open(input_path + labels_name, 'rb'))

    new_seqs = []
    new_labels = []
    for seq, label in zip(seqs, labels):
        if len(seq) < min_threshold or len(seq) >= max_threshold:
            continue
        else:
            new_seqs.append(seq)
            new_labels.append(label)
    seqs = new_seqs
    labels = new_labels

    seqs, labels = find_patients_with_many_associations(seqs, labels, association_threshold)

    temp_seqs, test_seqs, temp_labels, test_labels = train_test_split(seqs, labels, test_size=0.2, random_state=seed)
    train_seqs, valid_seqs, train_labels, valid_labels = train_test_split(temp_seqs, temp_labels, test_size=0.1, random_state=seed)

    train_size = int(len(train_seqs) * train_ratio)
    train_seqs = train_seqs[:train_size]
    train_labels = train_labels[:train_size]

    if filter_codes:
        dx_codes, rx_codes, pr_codes = build_dicts(train_seqs)
        valid_seqs = filter_unknown_codes(dx_codes, rx_codes, pr_codes, valid_seqs)
        test_seqs = filter_unknown_codes(dx_codes, rx_codes, pr_codes, test_seqs)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    sorted_index = len_argsort(train_seqs)
    train_seqs = [train_seqs[i] for i in sorted_index]
    train_labels = [train_labels[i] for i in sorted_index]

    sorted_index = len_argsort(valid_seqs)
    valid_seqs = [valid_seqs[i] for i in sorted_index]
    valid_labels = [valid_labels[i] for i in sorted_index]

    sorted_index = len_argsort(test_seqs)
    test_seqs = [test_seqs[i] for i in sorted_index]
    test_labels = [test_labels[i] for i in sorted_index]


    return train_seqs, train_labels, valid_seqs, valid_labels, test_seqs, test_labels


def find_patients_with_many_associations(seqs, labels, threshold):
    new_seqs = []
    new_labels = []
    for patient, label in zip(seqs, labels):
        code_count = 0
        dxobj_count = 0
        for visit in patient:
            local_code_count = 0
            for dxset in visit:
                dx = dxset[0]
                rxs = dxset[1]
                prs = dxset[2]
                if len(rxs) > 0 or len(prs) > 0:
                    local_code_count += 1
            if local_code_count > 1:
                code_count += 1
        num_visits = float(len(patient))
        if code_count / num_visits >= threshold:
            new_seqs.append(patient)   
            new_labels.append(label)
    return new_seqs, new_labels


def build_dicts(seqs):
    dx_codes = []
    rx_codes = []
    pr_codes = []

    for patient in seqs:
        for visit in patient:
            for dxobj in visit:
                dx_codes.append(dxobj[0])
                rx_codes.extend(dxobj[1])
                pr_codes.extend(dxobj[2])

    return set(dx_codes), set(rx_codes), set(pr_codes)

def filter_unknown_codes(dx_codes, rx_codes, pr_codes, seqs):
    new_seqs = []
    for patient in seqs:
        new_patient = []
        for visit in patient:
            new_visit = []
            for dxobj in visit:
                dx = dxobj[0]
                if dx not in dx_codes: continue
                new_rxs = [rx for rx in dxobj[1] if rx in rx_codes]
                new_prs = [pr for pr in dxobj[2] if pr in pr_codes]
                new_visit.append([dx, new_rxs, new_prs])
            new_patient.append(new_visit)
        new_seqs.append(new_patient)
    return new_seqs


def preprocess_hierarchical(patients, options):
    num_dx = options['num_dx']
    num_rx = options['num_rx']

    new_patients = []
    for i, patient in enumerate(patients):
        new_patient = []
        for j, visit in enumerate(patient):
            codes = []
            for k, diagnosis in enumerate(visit):
                dx_code = diagnosis[0]
                rx_codes = np.array(diagnosis[1]) + num_dx
                proc_codes = np.array(diagnosis[2]) + num_dx + num_rx
                codes.append(dx_code)
                codes.extend(list(rx_codes))
                codes.extend(list(proc_codes))
            new_patient.append(list(set(codes)))
        new_patients.append(new_patient)

    return new_patients


def sample_batch(seqs, labels, batch_size):
    idx = np.random.randint(0, len(seqs) - batch_size + 1)
    return seqs[idx:idx+batch_size], labels[idx:idx+batch_size]


def preprocess_hf(seqs, options):
    lengths = np.array([len(seq) for seq in seqs])
    max_length = np.max(lengths)
    num_samples = len(seqs)
    x = np.zeros((num_samples, max_length, options['input_size'])).astype('float32')
    #mask = np.zeros((num_samples, max_length)).astype('float32')

    for idx, seq in enumerate(seqs):
        for xvec, subseq in zip(x[idx], seq):
            xvec[subseq] = 1.
        #mask[idx, :lengths[idx]] = 1.

    x = np.transpose(x, (1, 0, 2)) #time-major RNN
    #mask = np.transpose(mask) #time-major RNN
    lengths = np.array(lengths).astype('int64')
    return x, lengths


def preprocess_dpm(seqs, labels, options):
    lengths = np.array([len(seq) for seq in seqs]) - 1
    max_length = np.max(lengths)
    num_samples = len(seqs)

    x = np.zeros((num_samples, max_length, options['input_size'])).astype('float32')
    y = np.zeros((num_samples, max_length, options['output_size'])).astype('float32')
    mask = np.zeros((num_samples, max_length)).astype('float32')

    for idx, (seq, lseq) in enumerate(zip(seqs, labels)):
        for xvec, subseq in zip(x[idx], seq[:-1]):
            xvec[subseq] = 1.
        for yvec, subseq in zip(y[idx], lseq[1:]):
            yvec[subseq] = 1.
        mask[idx, :lengths[idx]] = 1.

    x = np.transpose(x, (1, 0, 2)) #time-major RNN
    y = np.transpose(y, (1, 0, 2)) #time-major RNN
    mask = np.transpose(mask) #time-major RNN
    lengths = np.array(lengths).astype('int32')
    return x, y, mask, lengths


def count_parameters(exclude="rnn"):
    total_parameters = 0
    for variable in tf.trainable_variables():
        if exclude in str(variable.name): continue
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def recallTop(y_true, y_pred, rank=[5, 10, 20, 30]):
    recall = list()
    for i in range(len(y_pred)):
        thisOne = list()
        codes = y_true[i]
        tops = y_pred[i]
        for rk in rank:
            thisOne.append(len(set(codes).intersection(set(tops[:rk])))*1.0/len(set(codes)))
        recall.append( thisOne )
    return (np.array(recall)).mean(axis=0)


def preprocess_mp(seqs, labels, options):
    lengths = np.array([len(seq) for seq in seqs])
    max_length = np.max(lengths)
    num_samples = len(seqs)

    x = np.zeros((num_samples, max_length, options['input_size'])).astype('float32')
    y = np.zeros((num_samples, options['output_size'])).astype('float32')

    for idx, (seq, lseq) in enumerate(zip(seqs, labels)):
        for xvec, subseq in zip(x[idx], seq):
            xvec[subseq] = 1.
        y[idx][lseq] = 1.

    x = np.transpose(x, (1, 0, 2)) #time-major RNN
    lengths = np.array(lengths).astype('int32')
    return x, y, lengths


def st_preprocess_hf(patients, options):
    lengths = np.array([len(seq) for seq in patients])
    max_length = np.max(lengths)
    num_samples = len(patients)
    dx = np.zeros((num_samples, max_length, options['max_dx_per_visit'])).astype('int32')
    rx = np.zeros((num_samples, max_length, options['max_dx_per_visit'], options['max_rx_per_dx'])).astype('int32')
    pr = np.zeros((num_samples, max_length, options['max_dx_per_visit'], options['max_pr_per_dx'])).astype('int32')

    dx_mask = np.zeros((num_samples, max_length, options['max_dx_per_visit'])).astype('float32')
    rx_mask = np.zeros((num_samples, max_length, options['max_dx_per_visit'], options['max_rx_per_dx'])).astype('float32')
    pr_mask = np.zeros((num_samples, max_length, options['max_dx_per_visit'], options['max_pr_per_dx'])).astype('float32')

    for i, patient in enumerate(patients):
        for j, visit in enumerate(patient):
            for k, diagnosis in enumerate(visit):
                dx[i,j,k] = diagnosis[0]
                dx_mask[i,j,k] = 1.

                med_orders = list(diagnosis[1])
                rx[i,j,k][:len(med_orders)] = med_orders ####For now we only use codes. In the future, we may use extra info such as instructions
                rx_mask[i,j,k][:len(med_orders)] = 1.

                proc_orders = list(diagnosis[2])
                pr[i,j,k][:len(proc_orders)] = proc_orders ####For now we only use codes. In the future, we may use extra info such as components
                pr_mask[i,j,k][:len(proc_orders)] = 1.

    dx = np.transpose(dx, (1, 0, 2)) #time-major RNN
    rx = np.transpose(rx, (1, 0, 2, 3))
    pr = np.transpose(pr, (1, 0, 2, 3))
    dx_mask = np.transpose(dx_mask, (1, 0, 2))
    rx_mask = np.transpose(rx_mask, (1, 0, 2, 3))
    pr_mask = np.transpose(pr_mask, (1, 0, 2, 3))
    lengths = np.array(lengths).astype('int32')

    inputs = (dx, rx, pr)
    masks = (dx_mask, rx_mask, pr_mask)

    return inputs, masks, lengths


def st_preprocess_dpm(patients, labels, options):
    lengths = np.array([len(seq) for seq in patients]) - 1
    max_length = np.max(lengths)
    num_samples = len(patients)

    dx = np.zeros((num_samples, max_length, options['max_dx_per_visit'])).astype('int32')
    rx = np.zeros((num_samples, max_length, options['max_dx_per_visit'], options['max_rx_per_dx'])).astype('int32')
    pr = np.zeros((num_samples, max_length, options['max_dx_per_visit'], options['max_pr_per_dx'])).astype('int32')

    dx_mask = np.zeros((num_samples, max_length, options['max_dx_per_visit'])).astype('float32')
    rx_mask = np.zeros((num_samples, max_length, options['max_dx_per_visit'], options['max_rx_per_dx'])).astype('float32')
    pr_mask = np.zeros((num_samples, max_length, options['max_dx_per_visit'], options['max_pr_per_dx'])).astype('float32')

    y = np.zeros((num_samples, max_length, options['output_size'])).astype('float32')
    mask = np.zeros((num_samples, max_length)).astype('float32')

    for i, (patient, lseq) in enumerate(zip(patients, labels)):
        for j, visit in enumerate(patient[:-1]):
            for k, diagnosis in enumerate(visit):
                dx[i,j,k] = diagnosis[0]
                dx_mask[i,j,k] = 1.

                med_orders = diagnosis[1]
                rx[i,j,k][:len(med_orders)] = med_orders ####For now we only use codes. In the future, we may use extra info such as instructions
                rx_mask[i,j,k][:len(med_orders)] = 1.

                proc_orders = diagnosis[2]
                pr[i,j,k][:len(proc_orders)] = proc_orders ####For now we only use codes. In the future, we may use extra info such as components
                pr_mask[i,j,k][:len(proc_orders)] = 1.
        for yvec, subseq in zip(y[i], lseq[1:]):
            yvec[subseq] = 1.
        mask[i, :lengths[i]] = 1.

    dx = np.transpose(dx, (1, 0, 2)) #time-major RNN
    rx = np.transpose(rx, (1, 0, 2, 3))
    pr = np.transpose(pr, (1, 0, 2, 3))

    dx_mask = np.transpose(dx_mask, (1, 0, 2))
    rx_mask = np.transpose(rx_mask, (1, 0, 2, 3))
    pr_mask = np.transpose(pr_mask, (1, 0, 2, 3))
    mask = np.transpose(mask) #time-major RNN

    y = np.transpose(y, (1, 0, 2)) #time-major RNN
    lengths = np.array(lengths).astype('int32')

    inputs = (dx, rx, pr)
    masks = (dx_mask, rx_mask, pr_mask, mask)

    return inputs, y, masks, lengths


def st_preprocess_mp(patients, labels, options):
    lengths = np.array([len(seq) for seq in patients])
    max_length = np.max(lengths)
    num_samples = len(patients)

    dx = np.zeros((num_samples, max_length, options['max_dx_per_visit'])).astype('int32')
    dx_mask = np.zeros((num_samples, max_length, options['max_dx_per_visit'])).astype('float32')
    rx = np.zeros((num_samples, max_length, options['max_dx_per_visit'], options['max_rx_per_dx'])).astype('int32')
    rx_mask = np.zeros((num_samples, max_length, options['max_dx_per_visit'], options['max_rx_per_dx'])).astype('float32')
    pr = np.zeros((num_samples, max_length, options['max_dx_per_visit'], options['max_pr_per_dx'])).astype('int32')
    pr_mask = np.zeros((num_samples, max_length, options['max_dx_per_visit'], options['max_pr_per_dx'])).astype('float32')

    y = np.zeros((num_samples, options['output_size'])).astype('float32')

    for i, (patient, lseq) in enumerate(zip(patients, labels)):
        for j, visit in enumerate(patient):
            for k, diagnosis in enumerate(visit):
                dx[i,j,k] = diagnosis[0]
                dx_mask[i,j,k] = 1.

                med_orders = diagnosis[1]
                rx[i,j,k][:len(med_orders)] = med_orders ####For now we only use codes. In the future, we may use extra info such as instructions
                rx_mask[i,j,k][:len(med_orders)] = 1.

                proc_orders = diagnosis[2]
                pr[i,j,k][:len(proc_orders)] = proc_orders ####For now we only use codes. In the future, we may use extra info such as components
                pr_mask[i,j,k][:len(proc_orders)] = 1.
        y[i][lseq] = 1.

    dx = np.transpose(dx, (1, 0, 2)) #time-major RNN
    rx = np.transpose(rx, (1, 0, 2, 3))
    pr = np.transpose(pr, (1, 0, 2, 3))

    dx_mask = np.transpose(dx_mask, (1, 0, 2))
    rx_mask = np.transpose(rx_mask, (1, 0, 2, 3))
    pr_mask = np.transpose(pr_mask, (1, 0, 2, 3))

    lengths = np.array(lengths).astype('int32')

    inputs = (dx, rx, pr)
    masks = (dx_mask, rx_mask, pr_mask)

    return inputs, y, masks, lengths


def st_preprocess_hf_aux(patients, options):
    lengths = np.array([len(seq) for seq in patients])
    max_length = np.max(lengths)
    num_samples = len(patients)
    dx = np.zeros((num_samples, max_length, options['max_dx_per_visit'])).astype('int32')
    rx = np.zeros((num_samples, max_length, options['max_dx_per_visit'], options['max_rx_per_dx'])).astype('int32')
    pr = np.zeros((num_samples, max_length, options['max_dx_per_visit'], options['max_pr_per_dx'])).astype('int32')

    dx_mask = np.zeros((num_samples, max_length, options['max_dx_per_visit'])).astype('float32')
    rx_mask = np.zeros((num_samples, max_length, options['max_dx_per_visit'], options['max_rx_per_dx'])).astype('float32')
    pr_mask = np.zeros((num_samples, max_length, options['max_dx_per_visit'], options['max_pr_per_dx'])).astype('float32')

    dx_label = np.zeros((num_samples, max_length, options['max_dx_per_visit'], options['num_dx'])).astype('float32')
    rx_label = np.zeros((num_samples, max_length, options['max_dx_per_visit'], options['num_rx'])).astype('float32')
    pr_label = np.zeros((num_samples, max_length, options['max_dx_per_visit'], options['num_pr'])).astype('float32')

    for i, patient in enumerate(patients):
        for j, visit in enumerate(patient):
            for k, diagnosis in enumerate(visit):
                dx[i,j,k] = diagnosis[0]
                dx_mask[i,j,k] = 1.
                dx_label[i,j,k][diagnosis[0]] = 1.

                med_orders = list(diagnosis[1])
                rx[i,j,k][:len(med_orders)] = med_orders ####For now we only use codes. In the future, we may use extra info such as instructions
                rx_mask[i,j,k][:len(med_orders)] = 1.
                rx_label[i,j,k][med_orders] = 1.

                proc_orders = list(diagnosis[2])
                pr[i,j,k][:len(proc_orders)] = proc_orders ####For now we only use codes. In the future, we may use extra info such as components
                pr_mask[i,j,k][:len(proc_orders)] = 1.
                pr_label[i,j,k][proc_orders] = 1.

    dx = np.transpose(dx, (1, 0, 2)) #time-major RNN
    rx = np.transpose(rx, (1, 0, 2, 3))
    pr = np.transpose(pr, (1, 0, 2, 3))
    dx_mask = np.transpose(dx_mask, (1, 0, 2))
    rx_mask = np.transpose(rx_mask, (1, 0, 2, 3))
    pr_mask = np.transpose(pr_mask, (1, 0, 2, 3))
    dx_label = np.transpose(dx_label, (1, 0, 2, 3))
    rx_label = np.transpose(rx_label, (1, 0, 2, 3))
    pr_label = np.transpose(pr_label, (1, 0, 2, 3))
    lengths = np.array(lengths).astype('int32')

    inputs = (dx, rx, pr)
    masks = (dx_mask, rx_mask, pr_mask)
    labels = (dx_label, rx_label, pr_label)

    return inputs, labels, masks, lengths
