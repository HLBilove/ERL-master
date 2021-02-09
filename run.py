import numpy as np
import torch
import math
from sklearn import metrics
from utils import model_id_type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transpose_data_model = {'akt'}


def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log(np.maximum(1e-10, pred)) + \
        (1.0 - target) * np.log(np.maximum(1e-10, 1.0-pred))
    if mod == 'avg':
        return np.average(loss)*(-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False


def compute_auc(all_target, all_pred):
    #fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def train(net, params, optimizer, data, label):
    net.train()
    q_data = data[0]
    qa_data = data[1]
    pid_data = data[2]
    tid_data = data[3]
    fid_data = data[4]
    sd_data = data[5]
    rd_data = data[6]
    xid_data = data[7]
    yid_data = data[8]

    e_flag, p_flag, f_flag, a_flag, model_type = model_id_type(params.model)
    if not e_flag:
        params.n_pid = 0
    if not p_flag:
        params.n_tid = 0
        params.n_fid = 0
    if not f_flag:
        params.n_sd = 0
        params.n_rd = 0
    if not f_flag:
        params.n_xid = 0
        params.n_yid = 0

    N = int(math.ceil(len(q_data) / params.batch_size))
    q_data = q_data.T
    qa_data = qa_data.T
    shuffled_ind = np.arange(q_data.shape[1])
    np.random.shuffle(shuffled_ind)
    q_data = q_data[:, shuffled_ind]
    qa_data = qa_data[:, shuffled_ind]

    tid_data = tid_data.T
    tid_data = tid_data[:, shuffled_ind]
    fid_data = fid_data.T
    fid_data = fid_data[:, shuffled_ind]

    pid_data = pid_data.T
    pid_data = pid_data[:, shuffled_ind]
    sd_data = sd_data.T
    sd_data = sd_data[:, shuffled_ind]
    rd_data = rd_data.T
    rd_data = rd_data[:, shuffled_ind]
    xid_data = xid_data.T
    xid_data = xid_data[:, shuffled_ind]
    yid_data = yid_data.T
    yid_data = yid_data[:, shuffled_ind]

    pred_list = []
    target_list = []

    element_count = 0
    true_el = 0
    for idx in range(N):
        optimizer.zero_grad()
        q_one_seq = q_data[:, idx*params.batch_size:(idx+1)*params.batch_size]
        pid_one_seq = pid_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
        tid_one_seq = tid_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        fid_one_seq = fid_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        sd_one_seq = sd_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        rd_one_seq = rd_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        xid_one_seq = xid_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        yid_one_seq = yid_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]

        qa_one_seq = qa_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
        if model_type in transpose_data_model:
            input_q = np.transpose(q_one_seq[:, :])  # Shape (bs, seqlen)
            input_qa = np.transpose(qa_one_seq[:, :])  # Shape (bs, seqlen)
            target = np.transpose(qa_one_seq[:, :])
            input_pid = np.transpose(pid_one_seq[:, :])
            input_tid = np.transpose(tid_one_seq[:, :])
            input_fid = np.transpose(fid_one_seq[:, :])
            input_sd = np.transpose(sd_one_seq[:, :])
            input_rd = np.transpose(rd_one_seq[:, :])
            input_xid = np.transpose(xid_one_seq[:, :])
            input_yid = np.transpose(yid_one_seq[:, :])
        else:
            input_q = (q_one_seq[:, :])
            input_qa = (qa_one_seq[:, :])
            target = (qa_one_seq[:, :])
            input_pid = (pid_one_seq[:, :])
            input_tid = (tid_one_seq[:, :])
            input_fid = (fid_one_seq[:, :])
            input_sd = (sd_one_seq[:, :])
            input_rd = (rd_one_seq[:, :])
            input_xid = (xid_one_seq[:, :])
            input_yid = (yid_one_seq[:, :])
        target = (target - 1) / params.n_question
        target_1 = np.floor(target)
        el = np.sum(target_1 >= -.9)
        element_count += el

        input_q = torch.from_numpy(input_q).long().to(device)
        input_qa = torch.from_numpy(input_qa).long().to(device)
        target = torch.from_numpy(target_1).float().to(device)

        input_tid = torch.from_numpy(input_tid).long().to(device)
        input_fid = torch.from_numpy(input_fid).long().to(device)
        input_pid = torch.from_numpy(input_pid).long().to(device)
        input_sd = torch.from_numpy(input_sd).long().to(device)
        input_rd = torch.from_numpy(input_rd).long().to(device)
        input_xid = torch.from_numpy(input_xid).long().to(device)
        input_yid = torch.from_numpy(input_yid).long().to(device)

        loss, pred, true_ct = net(input_q, input_qa, target, input_pid, input_tid, input_fid,
                                  input_sd, input_rd, input_xid, input_yid)

        pred = pred.detach().cpu().numpy()
        loss.backward()
        true_el += true_ct.cpu().numpy()

        if params.maxgradnorm > 0.:
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), max_norm=params.maxgradnorm)

        optimizer.step()

        # correct: 1.0; wrong 0.0; padding -1.0
        target = target_1.reshape((-1,))

        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    acc = compute_accuracy(all_target, all_pred)

    return loss, acc, auc


def test(net, params, optimizer, data, label):
    q_data = data[0]
    qa_data = data[1]
    pid_data = data[2]
    tid_data = data[3]
    fid_data = data[4]
    sd_data = data[5]
    rd_data = data[6]
    xid_data = data[7]
    yid_data = data[8]

    e_flag, p_flag, f_flag, a_flag, model_type = model_id_type(params.model)
    if not e_flag:
        params.n_pid = 0
    if not p_flag:
        params.n_tid = 0
        params.n_fid = 0
    if not f_flag:
        params.n_sd = 0
        params.n_rd = 0
    if not f_flag:
        params.n_xid = 0
        params.n_yid = 0
    net.eval()

    N = int(math.ceil(float(len(q_data)) / float(params.batch_size)))
    q_data = q_data.T
    qa_data = qa_data.T
    pid_data = pid_data.T
    tid_data = tid_data.T
    fid_data = fid_data.T
    sd_data = sd_data.T
    rd_data = rd_data.T
    xid_data = xid_data.T
    yid_data = yid_data.T

    seq_num = q_data.shape[1]
    pred_list = []
    target_list = []

    count = 0
    true_el = 0
    element_count = 0
    for idx in range(N):
        q_one_seq = q_data[:, idx*params.batch_size:(idx+1)*params.batch_size]
        pid_one_seq = pid_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
        tid_one_seq = tid_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
        fid_one_seq = fid_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
        sd_one_seq = sd_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
        rd_one_seq = rd_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
        xid_one_seq = xid_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        yid_one_seq = yid_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]

        qa_one_seq = qa_data[:, idx * params.batch_size:(idx+1) * params.batch_size]

        if model_type in transpose_data_model:
            input_q = np.transpose(q_one_seq[:, :])
            input_qa = np.transpose(qa_one_seq[:, :])
            target = np.transpose(qa_one_seq[:, :])
            input_pid = np.transpose(pid_one_seq[:, :])
            input_tid = np.transpose(tid_one_seq[:, :])
            input_fid = np.transpose(fid_one_seq[:, :])
            input_sd = np.transpose(sd_one_seq[:, :])
            input_rd = np.transpose(rd_one_seq[:, :])
            input_xid = np.transpose(xid_one_seq[:, :])
            input_yid = np.transpose(yid_one_seq[:, :])
        else:
            input_q = (q_one_seq[:, :])
            input_qa = (qa_one_seq[:, :])
            target = (qa_one_seq[:, :])
            input_pid = (pid_one_seq[:, :])
            input_tid = (tid_one_seq[:, :])
            input_fid = (fid_one_seq[:, :])
            input_sd = (sd_one_seq[:, :])
            input_rd = (rd_one_seq[:, :])
            input_xid = (xid_one_seq[:, :])
            input_yid = (yid_one_seq[:, :])
        target = (target - 1) / params.n_question
        target_1 = np.floor(target)

        input_q = torch.from_numpy(input_q).long().to(device)
        input_qa = torch.from_numpy(input_qa).long().to(device)
        target = torch.from_numpy(target_1).float().to(device)
        input_pid = torch.from_numpy(input_pid).long().to(device)
        input_tid = torch.from_numpy(input_tid).long().to(device)
        input_fid = torch.from_numpy(input_fid).long().to(device)
        input_sd = torch.from_numpy(input_sd).long().to(device)
        input_rd = torch.from_numpy(input_rd).long().to(device)
        input_xid = torch.from_numpy(input_xid).long().to(device)
        input_yid = torch.from_numpy(input_yid).long().to(device)

        with torch.no_grad():
            loss, pred, ct = net(input_q, input_qa, target, input_pid, input_tid, input_fid,
                                 input_sd, input_rd, input_xid, input_yid)
        pred = pred.cpu().numpy()
        true_el += ct.cpu().numpy()
        if (idx + 1) * params.batch_size > seq_num:
            real_batch_size = seq_num - idx * params.batch_size
            count += real_batch_size
        else:
            count += params.batch_size

        target = target_1.reshape((-1,))
        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        element_count += pred_nopadding.shape[0]
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    assert count == seq_num, "Seq not matching"

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    acc = compute_accuracy(all_target, all_pred)

    return loss, acc, auc
