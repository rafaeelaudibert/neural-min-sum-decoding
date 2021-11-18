# Belief propagation using TensorFlow
# Run as follows:
# python3 main.py 0 1 6 1 100 10000000000000000 5 codes/BCH_63_45.alist codes/BCH_63_45.gmat 1.0 100 FNOMS
import numpy as np
import sys
from Decoder import Decoder
import constants

# Configure Tensorflow compatibility
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

seed = int(sys.argv[1])
np.random.seed(seed)

snr_lo = float(sys.argv[2])
snr_hi = float(sys.argv[3])
snr_step = float(sys.argv[4])
min_frame_errors = int(sys.argv[5])
max_frames = float(sys.argv[6])
num_iterations = int(sys.argv[7])
steps = int(sys.argv[11])
provided_decoder_type = sys.argv[12]

decoder = Decoder(decoder_type=provided_decoder_type, random_seed=1, relaxed=False)


# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
config = tf.ConfigProto(device_count={"CPU": 2, "GPU": 0})
with tf.Session(config=config) as session:  # tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    # simulate each SNR
    SNRs = np.arange(snr_lo, snr_hi + snr_step, snr_step)
    if (decoder.batch_size % len(SNRs)) != 0:
        print("********************")
        print("********************")
        print("error: batch size must divide by the number of SNRs to train on")
        print("********************")
        print("********************")
    BERs = []
    SERs = []
    FERs = []

    print("\nBuilding the decoder graph...")
    belief_propagation = decoder.belief_propagation_op(
        soft_input=decoder.tf_train_dataset, labels=decoder.tf_train_labels
    )
    if constants.TRAINING:
        training_loss = belief_propagation[
            5
        ]  # tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=belief_propagation[1], labels=tf_train_labels))
        loss = training_loss
        print("Learning rate: " + str(decoder.starter_learning_rate))
        optimizer = tf.train.AdamOptimizer(learning_rate=decoder.learning_rate).minimize(
            loss, global_step=decoder.global_step
        )
    print("Done.\n")
    init = tf.global_variables_initializer()

    if constants.ALL_ZEROS_CODEWORD_TRAINING:
        codewords = np.zeros([decoder.n, decoder.batch_size])
        codewords_repeated = np.zeros(
            [num_iterations, decoder.n, decoder.batch_size]
        )  # repeat for each iteration (multiloss)
        BPSK_codewords = np.ones([decoder.n, decoder.batch_size])
        soft_input = np.zeros_like(BPSK_codewords)
        channel_information = np.zeros_like(BPSK_codewords)

    covariance_matrix = np.eye(decoder.n)
    eta = 0.99
    for i in range(0, decoder.n):
        for j in range(0, decoder.n):
            covariance_matrix[i, j] = eta ** np.abs(i - j)

    session.run(init)

    if constants.TRAINING:
        print("***********************")
        print("Training decoder using " + str(steps) + " minibatches...")
        print("***********************")

        step = 0
        while step < steps:
            # generate random codewords
            if not constants.ALL_ZEROS_CODEWORD_TRAINING:
                # generate message
                messages = np.random.randint(0, 2, [decoder.k, decoder.batch_size])

                # encode message
                codewords = np.dot(decoder.G, messages) % 2
                # codewords_repeated = np.tile(x,(num_iterations,1,1)).shape

                # modulate codeword
                BPSK_codewords = (0.5 - codewords.astype(np.float32)) * 2.0

                soft_input = np.zeros_like(BPSK_codewords)
                channel_information = np.zeros_like(BPSK_codewords)
            else:
                codewords = np.zeros([decoder.n, decoder.batch_size])
                # codewords_repeated = np.zeros([num_iterations,n,batch_size]) # repeat for each iteration (multiloss)
                BPSK_codewords = np.ones([decoder.n, decoder.batch_size])
                soft_input = np.zeros_like(BPSK_codewords)
                channel_information = np.zeros_like(BPSK_codewords)

            # create minibatch with codewords from multiple SNRs
            for i in range(0, len(SNRs)):
                sigma = np.sqrt(1.0 / (2 * (np.float(decoder.k) / np.float(decoder.n)) * 10 ** (SNRs[i] / 10)))
                noise = sigma * np.random.randn(decoder.n, decoder.batch_size // len(SNRs))
                start_idx = decoder.batch_size * i // len(SNRs)
                end_idx = decoder.batch_size * (i + 1) // len(SNRs)
                channel_information[:, start_idx:end_idx] = BPSK_codewords[:, start_idx:end_idx] + noise
                if constants.NO_SIGMA_SCALING_TRAIN:
                    soft_input[:, start_idx:end_idx] = channel_information[:, start_idx:end_idx]
                else:
                    soft_input[:, start_idx:end_idx] = 2.0 * channel_information[:, start_idx:end_idx] / (sigma * sigma)

            # feed minibatch into BP and run SGD
            batch_data = soft_input
            batch_labels = codewords  # codewords #codewords_repeated
            feed_dict = {decoder.tf_train_dataset: batch_data, decoder.tf_train_labels: batch_labels}
            [_] = session.run(
                [optimizer], feed_dict=feed_dict
            )  # ,bp_output,syndrome_output,belief_propagation, soft_syndromes

            if decoder.relaxed and constants.TRAINING:
                print(session.run(decoder.R))

            if step % 100 == 0:
                print(str(step) + " minibatches completed")

            step += 1

        print("Trained decoder on " + str(step) + " minibatches.\n")
    else:
        saver = tf.train.Saver()
        saver.restore(session, constants.SAVE_PATH)

    # testing phase
    print("***********************")
    print("Testing decoder...")
    print("***********************")
    for SNR in SNRs:
        # simulate this SNR
        sigma = np.sqrt(1.0 / (2 * (np.float(decoder.k) / np.float(decoder.n)) * 10 ** (SNR / 10)))
        frame_count = 0
        bit_errors = 0
        frame_errors = 0
        frame_errors_with_HDD = 0
        symbol_errors = 0
        FE = 0

        # simulate frames
        while ((FE < min_frame_errors) or (frame_count < 100000)) and (frame_count < max_frames):
            frame_count += decoder.batch_size  # use different batch size for test phase?

            if not constants.ALL_ZEROS_CODEWORD_TESTING:
                # generate message
                messages = np.random.randint(0, 2, [decoder.batch_size, decoder.k])

                # encode message
                codewords = np.dot(decoder.G, messages.transpose()) % 2

                # modulate codeword
                BPSK_codewords = (0.5 - codewords.astype(np.float32)) * 2.0

            # add Gaussian noise to codeword
            noise = sigma * np.random.randn(BPSK_codewords.shape[0], BPSK_codewords.shape[1])
            channel_information = BPSK_codewords + noise

            # convert channel information to LLR format
            if constants.NO_SIGMA_SCALING_TEST:
                soft_input = channel_information
            else:
                soft_input = 2.0 * channel_information / (sigma * sigma)

            # run belief propagation
            batch_data = soft_input
            feed_dict = {decoder.tf_train_dataset: batch_data, decoder.tf_train_labels: codewords}
            soft_outputs = session.run([belief_propagation], feed_dict=feed_dict)
            soft_output = np.array(soft_outputs[0][1])
            recovered_codewords = (soft_output < 0).astype(int)

            # update bit error count and frame error count
            errors = codewords != recovered_codewords
            bit_errors += errors.sum()
            frame_errors += (errors.sum(0) > 0).sum()

            FE = frame_errors

        # summarize this SNR:
        print("SNR: " + str(SNR))
        print("frame count: " + str(frame_count))

        bit_count = frame_count * decoder.n
        BER = np.float(bit_errors) / np.float(bit_count)
        BERs.append(BER)
        print("bit errors: " + str(bit_errors))
        print("BER: " + str(BER))

        FER = np.float(frame_errors) / np.float(frame_count)
        FERs.append(FER)
        print("FER: " + str(FER))
        print("")

    # print summary
    print("BERs:")
    print(BERs)
    print("FERs:")
    print(FERs)

    if constants.TRAINING:
        saver = tf.train.Saver()
        saved_path = saver.save(session, constants.SAVE_PATH)
        print(f"Saved to {saved_path}")

    # offset = session.run(decoder.B_cv)
    # weights = session.run(decoder.W_cv)


# BERs:
# [0.09032932384251328, 0.06384019768312334, 0.03773727050105947, 0.017074118482991386, 0.005719234136500323, 0.0016139469376879449]
# FERs:
# [0.9644084732214229, 0.8444644284572342, 0.5838229416466827, 0.28393285371702637, 0.09335531574740208, 0.023521183053557153]
