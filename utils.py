import os
import torch
from erl_akt import ERL_AKT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def try_makedirs(path_):
    if not os.path.isdir(path_):
        try:
            os.makedirs(path_)
        except FileExistsError:
            pass

def get_file_name_identifier(params):
    words = params.model.split('_')
    model_type = words[0]
    if model_type == 'dkt':
        file_name = [['_b', params.batch_size], ['_gn', params.maxgradnorm], ['_lr', params.lr],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_dm', params.d_model], ['_ts', params.train_set],  ['_h', params.hidden_dim], ['_do', params.dropout], ['_l2', params.l2]]
    elif model_type == 'dktplus':
        file_name = [['_b', params.batch_size], ['_gn', params.maxgradnorm], ['_lr', params.lr],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_dm', params.d_model], ['_ts', params.train_set],  ['_h', params.hidden_dim], ['_do', params.dropout], ['_l2', params.l2], ['_r', params.lamda_r], ['_w1', params.lamda_w1], ['_w2', params.lamda_w2]]
    elif model_type == 'dkvmn':
        file_name = [['_b', params.batch_size], ['_gn', params.maxgradnorm], ['_lr', params.lr],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_q', params.q_embed_dim], ['_qa', params.qa_embed_dim], ['_ts', params.train_set], ['_m', params.memory_size], ['_l2', params.l2]]
    elif model_type in {'akt', 'sakt'}:
        file_name = [['_b', params.batch_size], ['_nb', params.n_block], ['_gn', params.maxgradnorm], ['_lr', params.lr],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_do', params.dropout], ['_dm', params.d_model], ['_ts', params.train_set], ['_kq', params.kq_same], ['_l2', params.l2]]
    return file_name


def model_id_type(model_name):
    words = model_name.split('_')
    is_e = True if 'e' in words else False
    is_p = True if 'p' in words else False
    is_f = True if 'f' in words else False
    is_a = True if 'a' in words else False
    return is_e, is_p, is_f, is_a, words[0]

def load_model(params):
    words = params.model.split('_')
    model_type = words[0]

    if model_type in {'akt'}:
        model = ERL_AKT(n_question=params.n_question, n_pid=params.n_pid, n_tid=params.n_tid, n_fid=params.n_fid,
                    n_sd=params.n_sd, n_rd=params.n_rd, n_xid=params.n_xid, n_yid=params.n_yid,
                    n_blocks=params.n_block, d_model=params.d_model, dropout=params.dropout, kq_same=params.kq_same,
                    model_type=model_type, l2=params.l2).to(device)
    else:
        model = None
    return model
