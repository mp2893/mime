import sys, pdb, os

import numpy as np
import cPickle as pickle
import tensorflow as tf
import sonnet as snt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize
import mime_util

def build_model(options):
    if options['emb_activation'] == 'sigmoid':
        emb_activation = tf.nn.sigmoid
    elif options['emb_activation'] == 'tanh':
        emb_activation = tf.nn.tanh
    else:
        emb_activation = tf.nn.relu

    if options['order_activation'] == 'sigmoid':
        order_activation = tf.nn.sigmoid
    elif options['order_activation'] == 'tanh':
        order_activation = tf.nn.tanh
    else:
        order_activation = tf.nn.relu

    if options['visit_activation'] == 'sigmoid':
        visit_activation = tf.nn.sigmoid
    elif options['visit_activation'] == 'tanh':
        visit_activation = tf.nn.tanh
    else:
        visit_activation = tf.nn.relu

    W_emb_dx = tf.get_variable('W_emb_dx', shape=(options['num_dx'], options['dx_emb_size']), dtype=tf.float32)
    W_emb_rx = tf.get_variable('W_emb_rx', shape=(options['num_rx'], options['rx_emb_size']), dtype=tf.float32)
    W_emb_pr = tf.get_variable('W_emb_pr', shape=(options['num_pr'], options['pr_emb_size']), dtype=tf.float32)

    dx_var = tf.placeholder(tf.int32, shape=(None, options['batch_size'], options['max_dx_per_visit']), name='dx_var')
    rx_var = tf.placeholder(tf.int32, shape=(None, options['batch_size'], options['max_dx_per_visit'], options['max_rx_per_dx']), name='rx_var')
    pr_var = tf.placeholder(tf.int32, shape=(None, options['batch_size'], options['max_dx_per_visit'], options['max_pr_per_dx']), name='pr_var')

    dx_label = tf.placeholder(tf.float32, shape=(None, options['batch_size'], options['max_dx_per_visit'], options['num_dx']), name='dx_label')
    rx_label = tf.placeholder(tf.float32, shape=(None, options['batch_size'], options['max_dx_per_visit'], options['num_rx']), name='rx_label')
    pr_label = tf.placeholder(tf.float32, shape=(None, options['batch_size'], options['max_dx_per_visit'], options['num_pr']), name='pr_label')

    dx_mask = tf.placeholder(tf.float32, shape=(None, options['batch_size'], options['max_dx_per_visit']), name='dx_mask')
    rx_mask = tf.placeholder(tf.float32, shape=(None, options['batch_size'], options['max_dx_per_visit'], options['max_rx_per_dx']), name='rx_mask')
    pr_mask = tf.placeholder(tf.float32, shape=(None, options['batch_size'], options['max_dx_per_visit'], options['max_pr_per_dx']), name='pr_mask')

    dx_visit = tf.nn.embedding_lookup(W_emb_dx, tf.reshape(dx_var, (-1, options['max_dx_per_visit'])))
    dx_visit = emb_activation(dx_visit)
    dx_visit = tf.reshape(dx_visit, (-1, options['dx_emb_size']))

    rx_visit = tf.nn.embedding_lookup(W_emb_rx, tf.reshape(rx_var, (-1, options['max_rx_per_dx'])))
    rx_visit = emb_activation(rx_visit)
    rx_visit = rx_visit * tf.reshape(rx_mask, (-1, options['max_rx_per_dx']))[:, :, None] ####Masking####
    rx_visit = tf.reduce_sum(rx_visit, axis=1)
    W_dr = snt.Sequential([snt.Linear(output_size=options['rx_emb_size'], name='W_dr'), order_activation])
    dr_visit = W_dr(dx_visit)
    dr_visit = dr_visit * rx_visit

    pr_visit = tf.nn.embedding_lookup(W_emb_pr, tf.reshape(pr_var, (-1, options['max_pr_per_dx'])))
    pr_visit = emb_activation(pr_visit)
    pr_visit = pr_visit * tf.reshape(pr_mask, (-1, options['max_pr_per_dx']))[:, :, None] ####Masking####
    pr_visit = tf.reduce_sum(pr_visit, axis=1)
    W_dp = snt.Sequential([snt.Linear(output_size=options['pr_emb_size'], name='W_dr'), order_activation])
    dp_visit = W_dp(dx_visit)
    dp_visit = dp_visit * pr_visit

    dx_obj = dx_visit + dr_visit + dp_visit
    W_dx = snt.Sequential([snt.Linear(output_size=options['dxobj_emb_size'], name='W_dxobj'), order_activation])
    dx_obj = W_dx(dx_obj)
    pre_visit = tf.reshape(dx_obj, (-1, options['max_dx_per_visit'], options['dxobj_emb_size']))
    pre_visit = pre_visit * tf.reshape(dx_mask, (-1, options['max_dx_per_visit']))[:, :, None] ####Masking####
    visit = tf.reduce_sum(pre_visit, axis=1)
    seq_visit = tf.reshape(visit, (-1, options['batch_size'], options['visit_emb_size']))

    seq_length = tf.placeholder(tf.int32, shape=(options['batch_size']), name='seq_length')
    rnn = snt.GRU(options['rnn_size'], name='emb2rnn')
    rnn2pred = snt.Sequential([snt.Linear(output_size=options['output_size'], name='rnn2pred'), tf.nn.sigmoid])
    rnn2aux_dx = snt.Linear(output_size=options['num_dx'], name='rnn2aux_dx')
    rnn2aux_rx = snt.Linear(output_size=options['num_rx'], name='rnn2aux_rx')
    rnn2aux_pr = snt.Linear(output_size=options['num_pr'], name='rnn2aux_pr')

    _, final_states = tf.nn.dynamic_rnn(rnn, seq_visit, dtype=tf.float32, time_major=True, sequence_length=seq_length)
    preds = tf.squeeze(rnn2pred(final_states))
    labels = tf.placeholder(tf.float32, shape=(options['batch_size']), name='labels')
    loss = -tf.reduce_mean(labels * tf.log(preds + 1e-10) + (1. - labels) * tf.log(1. - preds + 1e-10))

    aux_dx_preds = rnn2aux_dx(dx_obj) * tf.reshape(dx_mask, (-1, 1))
    aux_dx_loss = tf.losses.softmax_cross_entropy(tf.reshape(dx_label, (-1, options['num_dx'])), aux_dx_preds)

    aux_rx_preds = rnn2aux_rx(dx_obj) * tf.reshape(dx_mask, (-1, 1))
    aux_rx_loss = tf.losses.sigmoid_cross_entropy(tf.reshape(rx_label, (-1, options['num_rx'])), aux_rx_preds)

    aux_pr_preds = rnn2aux_pr(dx_obj) * tf.reshape(dx_mask, (-1, 1))
    aux_pr_loss = tf.losses.sigmoid_cross_entropy(tf.reshape(pr_label, (-1, options['num_pr'])), aux_pr_preds)

    input_tensors = (dx_var, rx_var, pr_var)
    label_tensors = (dx_label, rx_label, pr_label, labels)
    mask_tensors = (dx_mask, rx_mask, pr_mask)
    loss_tensors = (aux_dx_loss, aux_rx_loss, aux_pr_loss, loss)

    return input_tensors, label_tensors, mask_tensors, loss_tensors, seq_length, preds


def run_test(seqs, label_seqs, sess, preds_T, input_PHs, label_PHs, mask_PHs, seq_length_PH, loss_T, options):
    all_losses = []
    all_preds = []
    all_labels = []
    batch_size = options['batch_size']
    for idx in xrange(len(label_seqs) / batch_size):
        batch_x = seqs[idx*batch_size:(idx+1)*batch_size]
        batch_y = label_seqs[idx*batch_size:(idx+1)*batch_size]
        inputs, _, masks, seq_length = mime_util.st_preprocess_hf_aux(batch_x, options)
        preds, loss = sess.run([preds_T, loss_T],
                feed_dict={
                    input_PHs[0]:inputs[0],
                    input_PHs[1]:inputs[1],
                    input_PHs[2]:inputs[2],
                    mask_PHs[0]:masks[0],
                    mask_PHs[1]:masks[1],
                    mask_PHs[2]:masks[2],
                    label_PHs[-1]:batch_y,
                    seq_length_PH:seq_length,
                    }
                )
        all_losses.append(loss)
        all_preds.extend(list(preds))
        all_labels.extend(batch_y)
    auc = roc_auc_score(all_labels, all_preds)
    aucpr = average_precision_score(all_labels, all_preds)
    accuracy = (np.array(all_labels) == np.squeeze(binarize(np.array(all_preds).reshape(-1, 1), threshold=.5))).mean()
    return np.mean(all_losses), auc, aucpr


def train(
        input_path='',
        batch_size=100,
        num_iter=100,
        eval_period=10,
        num_eval=100,
        rnn_size=256,
        output_size=1,
        learning_rate=1e-3,
        output_path='',
        random_seed=1234,
        split_seed=1234,
        emb_activation='sigmoid',
        order_activation='sigmoid',
        visit_activation='sigmoid',
        num_dx=100,
        num_rx=100,
        num_pr=100,
        dx_emb_size=128,
        rx_emb_size=128,
        pr_emb_size=128,
        dxobj_emb_size=128,
        visit_emb_size=128,
        max_dx_per_visit=29,
        max_rx_per_dx=17,
        max_pr_per_dx=10,
        regularize=1e-3,
        aux_lambda=0.1,
        min_threshold=5,
        max_threshold=150,
        train_ratio=1.0,
        association_threshold=0.0,
        ):
    options = locals().copy()
    input_PHs, label_PHs, mask_PHs, loss_Ts, seq_length_PH, preds_T = build_model(options)

    all_vars = tf.trainable_variables()
    L2_loss = tf.constant(0.0, dtype=tf.float32)
    for var in all_vars:
        if len(var.shape) < 2:
            continue
        L2_loss += tf.reduce_sum(var ** 2)

    optimizer = tf.train.AdamOptimizer(learning_rate=options['learning_rate'])
    loss_T = options['aux_lambda'] * (loss_Ts[0] + loss_Ts[1] + loss_Ts[2]) + loss_Ts[3]
    minimize_op = optimizer.minimize(loss_T + regularize * L2_loss)
    train_seqs, train_labels, valid_seqs, valid_labels, test_seqs, test_labels = mime_util.load_data(
            options['input_path'],
            min_threshold=options['min_threshold'],
            max_threshold=options['max_threshold'],
            seed=options['split_seed'],
            train_ratio=options['train_ratio'],
            association_threshold=options['association_threshold'])

    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        best_valid_loss = 100000.0
        best_test_loss = 100000.0
        best_valid_auc = 0.0
        best_test_auc = 0.0
        best_valid_aucpr = 0.0
        best_test_aucpr = 0.0
        for train_iter in xrange(options['num_iter']+1):
            batch_x, batch_y = mime_util.sample_batch(train_seqs, train_labels, options['batch_size'])
            inputs, labels, masks, seq_length = mime_util.st_preprocess_hf_aux(batch_x, options)
            _, preds, losses = sess.run([minimize_op, preds_T, loss_Ts],
                    feed_dict={
                        input_PHs[0]:inputs[0],
                        input_PHs[1]:inputs[1],
                        input_PHs[2]:inputs[2],
                        mask_PHs[0]:masks[0],
                        mask_PHs[1]:masks[1],
                        mask_PHs[2]:masks[2],
                        label_PHs[0]:labels[0],
                        label_PHs[1]:labels[1],
                        label_PHs[2]:labels[2],
                        label_PHs[3]:batch_y,
                        seq_length_PH:seq_length,
                        }
                    )

            if train_iter > 0 and train_iter % options['eval_period'] == 0:
                valid_loss, valid_auc, valid_aucpr = run_test(valid_seqs, valid_labels, sess, preds_T, input_PHs, label_PHs, mask_PHs, seq_length_PH, loss_Ts[-1], options)
                if valid_loss < best_valid_loss:
                    test_loss, test_auc, test_aucpr = run_test(test_seqs, test_labels, sess, preds_T, input_PHs, label_PHs, mask_PHs, seq_length_PH, loss_Ts[-1], options)
                    best_valid_loss = valid_loss
                    best_valid_auc = valid_auc
                    best_valid_aucpr = valid_aucpr
                    best_test_loss = test_loss
                    best_test_auc = test_auc
                    best_test_aucpr = test_aucpr
                    savePath = saver.save(sess, output_path + '/r' + str(random_seed) + 's' + str(split_seed) + '/model', global_step=train_iter)
                print('round:%d, valid_loss:%f, valid_auc:%f, valid_aucpr:%f' % (train_iter, valid_loss, valid_auc, valid_aucpr))
        return best_valid_loss, best_test_loss, best_valid_auc, best_test_auc, best_valid_aucpr, best_test_aucpr


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    log_path = sys.argv[3]

    num_dx=388
    num_rx=99
    num_pr=1824
    rnn_size=256
    dx_emb_size=200
    rx_emb_size=dx_emb_size
    pr_emb_size=dx_emb_size
    dxobj_emb_size=256
    visit_emb_size=dxobj_emb_size
    max_dx_per_visit=22
    max_rx_per_dx=17
    max_pr_per_dx=10
    emb_activation='relu'
    order_activation='relu'
    visit_activation='relu'
    regularize=1e-4
    aux_lambda=0.0

    valid_losses = []
    test_losses = []
    valid_aucs = []
    test_aucs = []
    valid_aucprs = []
    test_aucprs = []
    for i in range(1):
        tf.set_random_seed(i)
        np.random.seed(i)
        for j in range(5):
            os.makedirs(output_path + '/r' + str(i) + 's' + str(j) + '/')
            tf.reset_default_graph()
            valid_loss, test_loss, valid_auc, test_auc, valid_aucpr, test_aucpr = train(
                    input_path=input_path,
                    output_path=output_path,
                    batch_size=20,
                    num_iter=20000,
                    eval_period=100,
                    rnn_size=rnn_size,
                    num_dx=num_dx,
                    num_rx=num_rx,
                    num_pr=num_pr,
                    dx_emb_size=dx_emb_size,
                    rx_emb_size=rx_emb_size,
                    pr_emb_size=pr_emb_size,
                    dxobj_emb_size=dxobj_emb_size,
                    visit_emb_size=visit_emb_size,
                    max_dx_per_visit=max_dx_per_visit,
                    max_rx_per_dx=max_rx_per_dx,
                    max_pr_per_dx=max_pr_per_dx,
                    emb_activation=emb_activation,
                    order_activation=order_activation,
                    visit_activation=visit_activation,
                    regularize=regularize,
                    aux_lambda=aux_lambda,
                    random_seed=i,
                    split_seed=j)
            valid_losses.append(valid_loss)
            test_losses.append(test_loss)
            valid_aucs.append(valid_auc)
            test_aucs.append(test_auc)
            valid_aucprs.append(valid_aucpr)
            test_aucprs.append(test_aucpr)
            buf  = "valid_loss:%f, test_loss:%f, valid_auc:%f, test_auc:%f, valid_aucpr:%f, test_aucpr:%f" % (valid_loss, test_loss, valid_auc, test_auc, valid_aucpr, test_aucpr)
            with open(log_path + '.log', 'a') as outfd: outfd.write(buf + '\n')
            print(buf)
    buf  = "mean_valid_loss:%f, mean_test_loss:%f, mean_valid_auc:%f, mean_test_auc:%f, mean_valid_aucpr:%f, mean_test_aucpr:%f" % (np.mean(valid_losses), np.mean(test_losses), np.mean(valid_aucs), np.mean(test_aucs), np.mean(valid_aucprs), np.mean(test_aucprs))
    with open(log_path + '.log', 'a') as outfd: outfd.write(buf + '\n')
    print(buf)
