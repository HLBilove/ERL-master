import numpy as np
import math
class DATA(object):
    def __init__(self, n_question, seqlen, separate_char, name="data"):
        self.separate_char = separate_char
        self.seqlen = seqlen
        self.n_question = n_question

    def load_data(self, path):
        file_data = open(path, 'r')
        q_data = []
        qa_data = []
        p_data = []
        t_data = []
        f_data = []
        sd_data = []
        rd_data = []
        x_data = []
        y_data = []
        for lineID, line in enumerate(file_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 6 == 0:
                learner_id = lineID//3
            if lineID % 6 == 1:
                SQ = line.split(self.separate_char)
                if len(SQ[len(SQ)-1]) == 0:
                    SQ = SQ[:-1]
                S = []
                Q = []
                for i in range(len(SQ)):
                    S.append(SQ[i].split(' ')[0])
                    Q.append(SQ[i].split(' ')[1])
            if lineID % 6 == 2:
                Performance = line.split(self.separate_char)
                if len(Performance[len(Performance)-1]) == 0:
                    Performance = Performance[:-1]
                T = []
                F = []
                for i in range(len(Performance)):
                    T.append(Performance[i].split(' ')[0])
                    F.append(Performance[i].split(' ')[1])
            if lineID % 6 == 3:
                Forget = line.split(self.separate_char)
                if len(Forget[len(Forget)-1]) == 0:
                    Forget = Forget[:-1]
                SD = []
                RD = []
                for i in range(len(Forget)):
                    SD.append(Forget[i].split(' ')[0])
                    if len(Forget[i].split(' ')) == 1:
                        RD.append(Forget[i].split(' ')[0])
                    else:
                        RD.append(Forget[i].split(' ')[1])
            if lineID % 6 == 4:
                SideFactors = line.split(self.separate_char)
                if len(SideFactors[len(SideFactors)-1]) == 0:
                    SideFactors = SideFactors[:-1]
                X = []
                Y = []
                for i in range(len(SideFactors)):
                    X.append(SideFactors[i].split(' ')[0])
                    if len(SideFactors[i].split(' ')) == 1:
                        Y.append(SideFactors[i].split(' ')[0])
                    else:
                        Y.append(SideFactors[i].split(' ')[1])
            if lineID % 6 == 5:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]

                # start split the data
                n_split = 1
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1

                for k in range(n_split):
                    q_seq = []
                    p_seq = []
                    t_seq = []
                    f_seq = []
                    sd_seq = []
                    rd_seq = []
                    x_seq = []
                    y_seq = []
                    a_seq = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0:
                            Xindex = int(S[i]) + round(float(A[i])) * self.n_question
                            q_seq.append(int(S[i]))
                            p_seq.append(int(Q[i]))
                            t_seq.append(int(T[i]))
                            f_seq.append(int(F[i]))
                            sd_seq.append(int(SD[i]))
                            rd_seq.append(int(RD[i]))
                            x_seq.append(int(X[i]))
                            y_seq.append(int(Y[i]))
                            a_seq.append(Xindex)
                        else:
                            print(Q[i])
                    q_data.append(q_seq)
                    qa_data.append(a_seq)
                    p_data.append(p_seq)
                    t_data.append(t_seq)
                    f_data.append(f_seq)
                    sd_data.append(sd_seq)
                    rd_data.append(rd_seq)
                    x_data.append(x_seq)
                    y_data.append(y_seq)

        file_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat

        p_dataArray = np.zeros((len(p_data), self.seqlen))
        for j in range(len(p_data)):
            dat = p_data[j]
            p_dataArray[j, :len(dat)] = dat

        t_dataArray = np.zeros((len(t_data), self.seqlen))
        for j in range(len(t_data)):
            dat = t_data[j]
            t_dataArray[j, :len(dat)] = dat

        f_dataArray = np.zeros((len(f_data), self.seqlen))
        for j in range(len(f_data)):
            dat = f_data[j]
            f_dataArray[j, :len(dat)] = dat

        sd_dataArray = np.zeros((len(sd_data), self.seqlen))
        for j in range(len(sd_data)):
            dat = sd_data[j]
            sd_dataArray[j, :len(dat)] = dat

        rd_dataArray = np.zeros((len(rd_data), self.seqlen))
        for j in range(len(rd_data)):
            dat = rd_data[j]
            rd_dataArray[j, :len(dat)] = dat

        x_dataArray = np.zeros((len(x_data), self.seqlen))
        for j in range(len(x_data)):
            dat = x_data[j]
            x_dataArray[j, :len(dat)] = dat

        y_dataArray = np.zeros((len(y_data), self.seqlen))
        for j in range(len(y_data)):
            dat = y_data[j]
            y_dataArray[j, :len(dat)] = dat

        return q_dataArray, qa_dataArray, p_dataArray, t_dataArray, f_dataArray, \
               sd_dataArray, rd_dataArray, x_dataArray, y_dataArray

    def load_test_data(self, path):
        file_data = open(path, 'r')
        q_data = []
        qa_data = []
        p_data = []
        t_data = []
        f_data = []
        sd_data = []
        rd_data = []
        x_data = []
        y_data = []
        test_q_num = 0
        for lineID, line in enumerate(file_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 6 == 0:
                learner_id = lineID//3
            if lineID % 6 == 1:
                SQ = line.split(self.separate_char)
                if len(SQ[len(SQ)-1]) == 0:
                    SQ = SQ[:-1]
                S = []
                Q = []
                for i in range(len(SQ)):
                    S.append(SQ[i].split(' ')[0])
                    Q.append(SQ[i].split(' ')[1])
                test_q_num += len(Q)
            if lineID % 6 == 2:
                Performance = line.split(self.separate_char)
                if len(Performance[len(Performance)-1]) == 0:
                    Performance = Performance[:-1]
                T = []
                F = []
                for i in range(len(Performance)):
                    T.append(Performance[i].split(' ')[0])
                    F.append(Performance[i].split(' ')[1])
            if lineID % 6 == 3:
                Forget = line.split(self.separate_char)
                if len(Forget[len(Forget)-1]) == 0:
                    Forget = Forget[:-1]
                SD = []
                RD = []
                for i in range(len(Forget)):
                    SD.append(Forget[i].split(' ')[0])
                    if len(Forget[i].split(' ')) == 1:
                        RD.append(Forget[i].split(' ')[0])
                    else:
                        RD.append(Forget[i].split(' ')[1])
            if lineID % 6 == 4:
                SideFactors = line.split(self.separate_char)
                if len(SideFactors[len(SideFactors)-1]) == 0:
                    SideFactors = SideFactors[:-1]
                X = []
                Y = []
                for i in range(len(SideFactors)):
                    X.append(SideFactors[i].split(' ')[0])
                    if len(SideFactors[i].split(' ')) == 1:
                        Y.append(SideFactors[i].split(' ')[0])
                    else:
                        Y.append(SideFactors[i].split(' ')[1])
            if lineID % 6 == 5:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]

                # start split the data
                n_split = 1
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1

                for k in range(n_split):
                    q_seq = []
                    p_seq = []
                    t_seq = []
                    f_seq = []
                    sd_seq = []
                    rd_seq = []
                    x_seq = []
                    y_seq = []
                    a_seq = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0:
                            Xindex = int(S[i]) + round(float(A[i])) * self.n_question
                            q_seq.append(int(S[i]))
                            p_seq.append(int(Q[i]))
                            t_seq.append(int(T[i]))
                            f_seq.append(int(F[i]))
                            sd_seq.append(int(SD[i]))
                            rd_seq.append(int(RD[i]))
                            x_seq.append(int(X[i]))
                            y_seq.append(int(Y[i]))
                            a_seq.append(Xindex)
                        else:
                            print(Q[i])
                    q_data.append(q_seq)
                    qa_data.append(a_seq)
                    p_data.append(p_seq)
                    t_data.append(t_seq)
                    f_data.append(f_seq)
                    sd_data.append(sd_seq)
                    rd_data.append(rd_seq)
                    x_data.append(x_seq)
                    y_data.append(y_seq)

        file_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat

        p_dataArray = np.zeros((len(p_data), self.seqlen))
        for j in range(len(p_data)):
            dat = p_data[j]
            p_dataArray[j, :len(dat)] = dat

        t_dataArray = np.zeros((len(t_data), self.seqlen))
        for j in range(len(t_data)):
            dat = t_data[j]
            t_dataArray[j, :len(dat)] = dat

        f_dataArray = np.zeros((len(f_data), self.seqlen))
        for j in range(len(f_data)):
            dat = f_data[j]
            f_dataArray[j, :len(dat)] = dat

        sd_dataArray = np.zeros((len(sd_data), self.seqlen))
        for j in range(len(sd_data)):
            dat = sd_data[j]
            sd_dataArray[j, :len(dat)] = dat

        rd_dataArray = np.zeros((len(rd_data), self.seqlen))
        for j in range(len(rd_data)):
            dat = rd_data[j]
            rd_dataArray[j, :len(dat)] = dat

        x_dataArray = np.zeros((len(x_data), self.seqlen))
        for j in range(len(x_data)):
            dat = x_data[j]
            x_dataArray[j, :len(dat)] = dat

        y_dataArray = np.zeros((len(y_data), self.seqlen))
        for j in range(len(y_data)):
            dat = y_data[j]
            y_dataArray[j, :len(dat)] = dat

        return q_dataArray, qa_dataArray, p_dataArray, t_dataArray, f_dataArray, \
               sd_dataArray, rd_dataArray, x_dataArray, y_dataArray, test_q_num