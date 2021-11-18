import numpy as np
import sys
from helper_functions import load_code, syndrome
import os
import constants

# Configure Tensorflow compatibility
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

np.set_printoptions(precision=3)

print("My PID: " + str(os.getpid()))

if constants.SUM_PRODUCT:
    print("Using Sum-Product algorithm")
if constants.MIN_SUM:
    print("Using Min-Sum algorithm")

if constants.ALL_ZEROS_CODEWORD_TRAINING:
    print("Training using only the all-zeros codeword")
else:
    print("Training using random codewords (not the all-zeros codeword)")

if constants.ALL_ZEROS_CODEWORD_TESTING:
    print("Testing using only the all-zeros codeword")
else:
    print("Testing using random codewords (not the all-zeros codeword)")

if constants.NO_SIGMA_SCALING_TRAIN:
    print("Not scaling train input by 2/sigma")
else:
    print("Scaling train input by 2/sigma")

if constants.NO_SIGMA_SCALING_TEST:
    print("Not scaling test input by 2/sigma")
else:
    print("Scaling test input by 2/sigma")

seed = int(sys.argv[1])
np.random.seed(seed)

num_iterations = int(sys.argv[7])
H_filename = sys.argv[8]
G_filename = sys.argv[9]
L = float(sys.argv[10])

if constants.ALL_ZEROS_CODEWORD_TESTING:
    G_filename = ""
code = load_code(H_filename, G_filename)


class Decoder:
    def __init__(self, decoder_type="RNOMS", random_seed=0, learning_rate=0.001, relaxed=False):

        # code.H = np.array([[1, 1, 0, 1, 1, 0, 0],
        #        [1, 0, 1, 1, 0, 1, 0],
        #        [0, 1, 1, 1, 0, 0, 1]])

        self.H = code.H
        self.G = code.G
        self.var_degrees = code.var_degrees
        self.chk_degrees = code.chk_degrees
        self.num_edges = code.num_edges
        self.u = code.u
        self.d = code.d
        self.n = code.n
        self.m = code.m
        self.k = code.k

        self.decoder_type = decoder_type
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.relaxed = relaxed

        # decoder parameters
        self.batch_size = 120
        self.tf_train_dataset = tf.placeholder(tf.float32, shape=(self.n, self.batch_size))
        self.tf_train_labels = tf.placeholder(
            tf.float32, shape=(self.n, self.batch_size)
        )  # tf.placeholder(tf.float32, shape=(num_iterations,n,batch_size))
        self.configure_decoder()

    # compute messages from variable nodes to check nodes
    def compute_vc(self, cv, iteration, soft_input):
        weighted_soft_input = soft_input

        edges = []
        for i in range(0, self.n):
            for j in range(0, self.var_degrees[i]):
                edges.append(i)
        reordered_soft_input = tf.gather(weighted_soft_input, edges)

        vc = []
        edge_order = []
        for i in range(0, self.n):  # for each variable node v
            for j in range(0, self.var_degrees[i]):
                # edge = d[i][j]
                edge_order.append(self.d[i][j])
                extrinsic_edges = []
                for jj in range(0, self.var_degrees[i]):
                    if jj != j:  # extrinsic information only
                        extrinsic_edges.append(self.d[i][jj])
                # if the list of edges is not empty, add them up
                if extrinsic_edges:
                    temp = tf.gather(cv, extrinsic_edges)
                    temp = tf.reduce_sum(temp, 0)
                else:
                    temp = tf.zeros([self.batch_size])
                if constants.SUM_PRODUCT:
                    temp = tf.cast(temp, tf.float32)  # tf.cast(temp, tf.float64)
                vc.append(temp)

        vc = tf.stack(vc)
        new_order = np.zeros(self.num_edges).astype(np.int)
        new_order[edge_order] = np.array(range(0, self.num_edges)).astype(np.int)
        vc = tf.gather(vc, new_order)
        vc = vc + reordered_soft_input
        return vc

    # compute messages from check nodes to variable nodes
    def compute_cv(self, vc, iteration):
        cv_list = []
        prod_list = []
        min_list = []

        if constants.SUM_PRODUCT:
            vc = tf.clip_by_value(vc, -10, 10)
            tanh_vc = tf.tanh(vc / 2.0)
        edge_order = []
        for i in range(0, self.m):  # for each check node c
            for j in range(0, self.chk_degrees[i]):
                # edge = u[i][j]
                edge_order.append(self.u[i][j])
                extrinsic_edges = []
                for jj in range(0, self.chk_degrees[i]):
                    if jj != j:
                        extrinsic_edges.append(self.u[i][jj])
                if constants.SUM_PRODUCT:
                    temp = tf.gather(tanh_vc, extrinsic_edges)
                    temp = tf.reduce_prod(temp, 0)
                    temp = tf.log((1 + temp) / (1 - temp))
                    cv_list.append(temp)
                if constants.MIN_SUM:
                    temp = tf.gather(vc, extrinsic_edges)
                    temp1 = tf.reduce_prod(tf.sign(temp), 0)
                    temp2 = tf.reduce_min(tf.abs(temp), 0)
                    prod_list.append(temp1)
                    min_list.append(temp2)

        if constants.SUM_PRODUCT:
            cv = tf.stack(cv_list)
        if constants.MIN_SUM:
            prods = tf.stack(prod_list)
            mins = tf.stack(min_list)
            if self.decoder_type == "RNOMS":
                # offsets = tf.nn.softplus(decoder.B_cv)
                # mins = tf.nn.relu(mins - tf.tile(tf.reshape(offsets,[-1,1]),[1,self.batch_size]))
                mins = tf.nn.relu(mins - self.B_cv)
            elif self.decoder_type == "FNOMS":
                offsets = tf.nn.softplus(self.B_cv[iteration])
                mins = tf.nn.relu(mins - tf.tile(tf.reshape(offsets, [-1, 1]), [1, self.batch_size]))
            cv = prods * mins

        new_order = np.zeros(self.num_edges).astype(np.int)
        new_order[edge_order] = np.array(range(0, self.num_edges)).astype(np.int)
        cv = tf.gather(cv, new_order)

        if self.decoder_type == "RNSPA" or self.decoder_type == "RNNMS":
            cv = cv * tf.tile(tf.reshape(self.W_cv, [-1, 1]), [1, self.batch_size])
        elif self.decoder_type == "FNSPA" or self.decoder_type == "FNNMS":
            cv = cv * tf.tile(tf.reshape(self.W_cv[iteration], [-1, 1]), [1, self.batch_size])
        return cv

    # combine messages to get posterior LLRs
    def marginalize(self, soft_input, iteration, cv):
        weighted_soft_input = soft_input

        soft_output = []
        for i in range(0, self.n):
            edges = []
            for e in range(0, self.var_degrees[i]):
                edges.append(self.d[i][e])

            temp = tf.gather(cv, edges)
            temp = tf.reduce_sum(temp, 0)
            soft_output.append(temp)

        soft_output = tf.stack(soft_output)

        soft_output = weighted_soft_input + soft_output
        return soft_output

    def continue_condition(self, soft_input, soft_output, iteration, cv, m_t, loss, labels):
        condition = iteration < num_iterations
        return condition

    def belief_propagation_iteration(self, soft_input, soft_output, iteration, cv, m_t, loss, labels):
        # compute vc
        vc = self.compute_vc(cv, iteration, soft_input)

        # filter vc
        if self.relaxed:
            m_t = self.R * m_t + (1 - self.R) * vc
            vc_prime = m_t
        else:
            vc_prime = vc

        # compute cv
        cv = self.compute_cv(vc_prime, iteration)

        # get output for this iteration
        soft_output = self.marginalize(soft_input, iteration, cv)
        iteration += 1

        # L = 0.5
        print("L = " + str(L))
        CE_loss = (
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=-soft_output, labels=labels)) / num_iterations
        )
        syndrome_loss = tf.reduce_mean(tf.maximum(1.0 - syndrome(soft_output, code), 0)) / num_iterations
        new_loss = L * CE_loss + (1 - L) * syndrome_loss
        loss = loss + new_loss

        return soft_input, soft_output, iteration, cv, m_t, loss, labels

    # builds a belief propagation TF graph
    def belief_propagation_op(self, soft_input, labels):
        return tf.while_loop(
            self.continue_condition,  # iteration < max iteration?
            self.belief_propagation_iteration,  # compute messages for this iteration
            [
                soft_input,  # soft input for this iteration
                soft_input,  # soft output for this iteration
                tf.constant(0, dtype=tf.int32),  # iteration number
                tf.zeros([self.num_edges, self.batch_size], dtype=tf.float32),  # cv
                tf.zeros([self.num_edges, self.batch_size], dtype=tf.float32),  # m_t
                tf.constant(0.0, dtype=tf.float32),  # loss
                labels,
            ],
        )

    def configure_decoder(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.starter_learning_rate = 0.01
        self.learning_rate = self.starter_learning_rate  # provided_decoder_type="normal", "FNNMS", "FNOMS", ...
        print("\n\nDecoder type: " + self.decoder_type + "\n\n")
        if self.relaxed:
            print("relaxed")
        else:
            print("not relaxed")

        if constants.SUM_PRODUCT:
            if self.decoder_type == "FNSPA":
                self.W_cv = tf.Variable(
                    tf.truncated_normal(
                        [num_iterations, self.num_edges], dtype=tf.float32, stddev=1.0, seed=self.random_seed
                    )
                )

            if self.decoder_type == "RNSPA":
                self.W_cv = tf.Variable(
                    tf.truncated_normal([self.num_edges], dtype=tf.float32, stddev=1.0, seed=self.random_seed)
                )  # tf.Variable(0.0,dtype=tf.float32)#

        if constants.MIN_SUM:
            if self.decoder_type == "FNNMS":
                # self.W_cv = tf.nn.softplus(tf.Variable(tf.truncated_normal([num_iterations, self.num_edges],dtype=tf.float32,stddev=1.0, seed=self.random_seed)))
                self.W_cv = tf.Variable(
                    tf.truncated_normal(
                        [num_iterations, self.num_edges], dtype=tf.float32, stddev=1.0, seed=self.random_seed
                    )
                )

            if self.decoder_type == "FNOMS":
                self.B_cv = tf.Variable(
                    tf.truncated_normal([num_iterations, self.num_edges], dtype=tf.float32, stddev=1.0)
                )  # tf.Variable(1.0 + tf.truncated_normal([num_iterations, self.num_edges],dtype=tf.float32,stddev=1.0))#tf.Variable(1.0 + tf.truncated_normal([num_iterations, self.num_edges],dtype=tf.float32,stddev=1.0/self.num_edges))

            if self.decoder_type == "RNNMS":
                self.W_cv = tf.nn.softplus(
                    tf.Variable(
                        tf.truncated_normal([self.num_edges], dtype=tf.float32, stddev=1.0, seed=self.random_seed)
                    )
                )  # tf.Variable(0.0,dtype=tf.float32)#

            if self.decoder_type == "RNOMS":
                self.B_cv = tf.Variable(
                    tf.truncated_normal([self.num_edges], dtype=tf.float32, stddev=1.0)
                )  # tf.Variable(0.0,dtype=tf.float32)#

        if self.relaxed:
            self.relaxation_factors = tf.Variable(0.0, dtype=tf.float32)
            R = tf.sigmoid(self.relaxation_factors)
            # print "single learned relaxation factor"

            # self.relaxation_factors = tf.Variable(tf.truncated_normal([self.num_edges],dtype=tf.float32,stddev=1.0))
            # R = tf.tile(tf.reshape(tf.sigmoid(self.relaxation_factors),[-1,1]),[1,batch_size])
            # print "multiple relaxation factors"
