import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import librosa

gpu_options = tf.GPUOptions(allow_growth = True)
config=tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.per_process_gpu_memory_fraction = 0.34

Sampling_Rate = 16000
Total_Files = 1200
batch_size = 10
Size_Frame = 513


def Train_Load_Files(path, Training_Files_Initial):
    print("Loading...")

    STFT_ABS = []
    Lengths = []
    File_Name_Extension = ['000', '00', '0', '']

    for i in range(Total_Files):
        if (i == 0):
            j = 0
        else:
            j = int(math.log10(i))
        # Using Librosa to load file np array from path
        Signal, Sampling_Rate = librosa.load(path + Training_Files_Initial + File_Name_Extension[j] + str(i) + '.wav',
                                             sr=None)

        # Using librosa to calculate stft
        Signal_STFT = librosa.stft(Signal, n_fft=1024, hop_length=512)
        Signal_STFT_Length = Signal_STFT.shape[1]

        # Abosulte values of STFT
        stft_abs = np.abs(Signal_STFT)

        STFT_ABS.append(stft_abs)

        # All stft lengths will be added to to the list STFT_ABS
        Lengths.append(Signal_STFT_Length)

    return STFT_ABS, Lengths



#Loading all signals and storing their stft absolutes and lengths.

Training_Files_Path = "/opt/e533/timit-homework/tr/"
Train_Mixed_Abs, Train_Mixed_Length = Train_Load_Files(Training_Files_Path, 'trx')
Train_Clean_Abs = Train_Load_Files(Training_Files_Path, 'trs')[0]
N_abs = Train_Load_Files(Training_Files_Path, 'trn')[0]

def IBM(clean, noise):
    Mask = []
    for i in range(len(clean)):
        temp = 1 * (clean[i] > noise[i])
        Mask.append(temp)
    return Mask

M = IBM(Train_Clean_Abs, N_abs)



def IBM(clean, noise):
    Mask = []
    for i in range(len(clean)):
        temp = 1 * (clean[i] > noise[i])
        Mask.append(temp)
    return Mask

Hidden_Units = 128


xs = tf.placeholder(tf.float32, [None, None, Size_Frame])
ys = tf.placeholder(tf.float32, [None, None, Size_Frame])


output1, state1 = tf.nn.dynamic_rnn(tf.contrib.rnn.LSTMCell(Hidden_Units, initializer = tf.contrib.layers.xavier_initializer()),
                                  xs, dtype=tf.float32)

output_final = tf.layers.dense(output1, 513, kernel_initializer= tf.contrib.layers.xavier_initializer())


fin_out = tf.sigmoid(output_final)
cost = tf.reduce_mean(tf.losses.mean_squared_error(fin_out, ys))
optimizer = tf.train.AdamOptimizer(learning_rate= 0.001).minimize(cost)


# Session started.
sess = tf.Session(config = config)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

error = np.zeros(100)
print("in train")
for epoch in range(100):
    randomize = np.arange(0, Total_Files, batch_size)
    np.random.shuffle(randomize)

    for i in range(len(randomize)):
        beg = int(randomize[i])
        end = int(beg + batch_size)
        epoch_y = np.array(M[beg:end]).swapaxes(1, 2)
        epoch_x = np.array(Train_Mixed_Abs[beg:end]).swapaxes(1, 2)

        l, _ = sess.run([cost, optimizer], feed_dict={xs: epoch_x, ys: epoch_y})
        error[epoch] += l

    print('Epoch', epoch + 1, 'completed out of ', 100, '; loss: ', error[epoch])

saver.save(sess, 'Model_question/q2')

saver.restore(sess, 'Model_question/q2')


# VALIDATION

def Validation_Load_File(path, Validation_File_Initials):
    print("Loading...")

    File_Name_Extension = ['000', '00', '0', '']
    Signals = []
    STFTs = []
    STFT_ABS = []
    Length = []

    for i in range(Total_Files):
        if (i == 0):
            j = 0
        else:
            j = int(math.log10(i))

        # Using Librosa to load file np array from path
        Signal, Sampling_Rate = librosa.load(path + Validation_File_Initials + File_Name_Extension[j] + str(i) + '.wav',
                                             sr=None)
        Signals.append(Signal)

        # stft is calculated using librosa and its length is also stored.
        Signal_STFT = librosa.stft(Signal, n_fft=1024, hop_length=512)
        Signal_STFT_Length = Signal_STFT.shape[1]

        STFTs.append(Signal_STFT)

        # Absolute value of stft
        Absolute_STFT = np.abs(Signal_STFT)

        # Adding STFTs to list
        STFT_ABS.append(Absolute_STFT)

        # Adding lengths to list
        Length.append(Signal_STFT_Length)

    return Signals, STFTs, STFT_ABS, Length


path = "/opt/e533/timit-homework/v/"

Validation_Mixed_Signals, Validation_Mixed_STFTs, Validation_Mixed_Absolute, Validation_Mixed_Length = Validation_Load_File(path, 'vx')
Validation_Clean_Signals, Validation_Clean_STFTs, Validation_Clean_Absolute, Validation_Clean_Length = Validation_Load_File(path, 'vs')
Validation_Noise_Signals, Validation_Noise_STFTs, Validation_Noise_Absolute, Validation_Noise_Length = Validation_Load_File(path, 'vn')

VM = IBM(Validation_Clean_Absolute, Validation_Noise_Absolute)

SNR_Val = np.zeros(Total_Files)


def SNR_Validation(M_p, X, s, i):
    M_p = 1 * (M_p > 0.5)
    S_p = M_p * X
    s_p = librosa.istft(S_p, win_length=1024, hop_length=512)

    lens = len(s)
    lensp = len(s_p)
    nlen = min(lens, lensp)

    SNR = 10 * math.log10((np.sum(s[:nlen] ** 2)) / (np.sum((s[:nlen] - s_p[:nlen]) ** 2)))

    return SNR


for i in range(len(Validation_Mixed_Absolute)):
    epoch_x = np.zeros((1, Validation_Mixed_Absolute[i].shape[1], Validation_Mixed_Absolute[i].shape[0]))
    epoch_y = np.zeros((1, Validation_Mixed_Absolute[i].shape[1], Validation_Mixed_Absolute[i].shape[0]))

    epoch_x[0, :, :] = Validation_Mixed_Absolute[i].T
    epoch_y[0, :, :] = Validation_Clean_Absolute[i].T

    VM_pred, val_loss = sess.run([fin_out, cost], feed_dict={xs: epoch_x, ys: epoch_y})

    SNR_Val[i] = SNR_Validation(VM_pred[0, :Validation_Mixed_Length[i], :].T, Validation_Mixed_STFTs[i],
                                Validation_Clean_Signals[i], i)

print(np.mean(SNR_Val))


path = "/opt/e533/timit-homework/te/"

Total_Files = 400

Test_Signals, Test_Signals_Stft, Test_Signals_Abs, Test_Signals_Length = Validation_Load_File(path, 'tex')


# SNR for test set
def Save_Test(M_pred, X, i):
    M_pred = 1 * (M_pred > 0.5)
    S_pred = M_pred * X
    s_pred = librosa.istft(S_pred, win_length=1024, hop_length=512)

    librosa.output.write_wav('Signals/tes_recover_' + str(i) + '.wav', s_pred, Sampling_Rate)


for i in range(len(Test_Signals_Abs)):
    epoch_x = np.zeros((1, Test_Signals_Abs[i].shape[1], Test_Signals_Abs[i].shape[0]))
    epoch_y = np.zeros((1, Test_Signals_Abs[i].shape[1], Test_Signals_Abs[i].shape[0]))

    epoch_x[0, :, :] = Test_Signals_Abs[i].T

    # Predictions for test signals
    TEM_pred = sess.run(fin_out, feed_dict={xs: epoch_x})

    # Output test signals
    Save_Test(TEM_pred[0, :, :].T, Test_Signals_Stft[i], i)

sess.close()
