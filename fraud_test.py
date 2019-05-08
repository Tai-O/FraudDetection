
#https://www.datascience.com/blog/fraud-detection-with-tensorflow



import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from keras import backend as K

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

#load data
df = pd.read_csv("creditcard_sample.csv")

#count number of fraud (1) and normal(0) rows of data
print(pd.value_counts(df['Class'], sort = True))


#preprocessing data
from sklearn.preprocessing import StandardScaler

data = df.drop(['Time'], axis=1)

data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
X_train = X_train[X_train.Class == 0]
X_train = X_train.drop(['Class'], axis=1)

y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)

X_train = X_train.values
X_test = X_test.values

#NN model 

'''
A Neural Network model namely Autoencoder is built to perform fraud detection/anomaly detection as a means of a unsupervised learning problem
perfomed using clustering
'''
input_dim = X_train.shape[1]
encoding_dim = 14


#encoder
'''
build encoder model's framework 
set dimension to the size of the training data
using L1 regularization to prevent overfitting
TanH for the neuron activation function
'''

'''
4. replace line 106 from l1 to l2 as it is better suited for clustering
'''
input_tensor = Input(shape=(input_dim, ))

encoderOut = Dense(encoding_dim, activation="tanh", 
                activity_regularizer=regularizers.l1(10e-5))(input_tensor)
encoderOut = Dense(int(encoding_dim / 2), activation="relu")(encoderOut)

encoder = Model(input_tensor, encoderOut)


#decoder
'''
build decoder model's framework 
set dimension to half the size of the training data
TanH for the hidden layer activation function
relu for final layer activation function
'''
decoder_input = Input(shape=(int(encoding_dim / 2),))
decoderOut = Dense(int(encoding_dim / 2), activation='tanh',name='decoder_input')(decoder_input)
decoderOut = Dense(input_dim, activation='relu',name='decoder_output')(decoderOut)

decoder = Model(decoder_input, decoderOut)

#autoencoder
autoInput = Input(shape=(input_dim, ))
encoderOut = encoder(autoInput)
decoderOut = decoder(encoderOut)
autoencoder = Model(inputs=autoInput, outputs=decoderOut)

#train
nb_epoch = 100
batch_size = 128


'''
performance metric for autoencoder
optimize weights/learning rates using Adam Optimizer
loss function of the model using mean square error
'''
autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)

tb = TensorBoard(log_dir='./logs',
                histogram_freq=0,
                write_graph=True,
                write_images=True)


history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer]).history


autoencoder = load_model('model.h5')


'''
model evaluation function to calculate model loss
'''

plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


'''
Reconstruction Error Check 
'''

test_x_predictions = autoencoder.predict(test_x)
mse = np.mean(np.power(test_x - test_x_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': test_y})
error_df.describe()


'''
ROC Curve Check 
'''

false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
roc_auc = auc(false_pos_rate, true_pos_rate,)

plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
plt.plot([0,1],[0,1], linewidth=5)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


'''
Recall vs. Precision Thresholding
'''
precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
plt.plot(recall_rt, precision_rt, linewidth=5, label='Precision-Recall curve')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=5)
plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()

'''
Reconstruction Error vs Threshold Check
'''

threshold_fixed = 5
groups = error_df.groupby('True_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Fraud" if name == 1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();


'''
Confusion Matrix
'''

pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.True_class, pred_y)

plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

