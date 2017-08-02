import numpy as np
import pdb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
np.random.seed(1006)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import optimizers
from keras.layers.normalization import BatchNormalization as bn
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import f1_score
from keras import metrics
from keras.utils.vis_utils import plot_model
from sklearn import decomposition
import matplotlib.pyplot as plt

def f1(y_true,y_pred):
	return f1_score(y_true,y_pred,average=None)

batch_size = 5 
epochs = 20

data = load_iris()
X = data.data
Y = keras.utils.to_categorical(data.target)
label_names = data.target_names.tolist()

labels = data.target_names
X_tr, X_te, y_tr, y_te = train_test_split(X,Y,test_size=0.2,random_state=1006)
n_train = X_tr.shape[0]
n_test = X_te.shape[0]
steps_per_epoch = n_train/batch_size

# Quick vis
# train test distrbution for y
y_tr_ = y_tr.argmax(1).flatten()
y_te_ = y_te.argmax(1).flatten()
bins = 3
plt.hist(y_tr_,bins=3,ec='black',alpha=0.5,label="Train distribution")
plt.hist(y_te_,bins=3,ec='black',alpha=0.5,label="Test distribution")
plt.legend()

# Pca XY plot
plt.savefig("tr_te_dist.png")
pca = decomposition.PCA(n_components=2)
pca.fit(X_tr)
X_tr_pc = pca.transform(X_tr)
plt.scatter(X_tr_pc[:,0],X_tr_pc[:,1],c=y_tr.argmax(1))
plt.savefig("vis.png")

def load_data(train_test,batch_size):
	if train_test == "train":
		X,Y = X_tr, y_tr 
	else:
		X,Y = X_te, y_te
	frm = 0
	to = frm + batch_size
	n = X.shape[0]
	while True:
		if frm >= n:
			frm = 0 
			to = frm + batch_size
		x = X[frm:to]
		y = Y[frm:to]
		frm = min(frm+batch_size,n)
		to = min(to+batch_size,n)
		yield (x, y)
		
model = Sequential()
model.add(Dense(10,input_shape=(4,)))
model.add(bn())
model.add(Activation('relu'))
model.add(Dense(4))
model.add(bn())
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))
adam = optimizers.Adam(lr=0.01)
model.compile(loss='categorical_crossentropy',
		optimizer=adam,
		metrics=['accuracy'])
plot_model(model,show_shapes=True,to_file='model.png',rankdir='TB')
model.summary()
model.save('weights.h5')

model.fit_generator(load_data("train",batch_size),
		    steps_per_epoch=steps_per_epoch,
		    nb_epoch =  epochs,
		    verbose=2)


score = model.evaluate(X_te,y_te,
			batch_size=batch_size,
			verbose=1)


pred_tr = model.predict(X_tr).argmax(1)
pred_te = model.predict(X_te).argmax(1)
y_tr = y_tr.argmax(1)
y_te = y_te.argmax(1)
cm_tr = cm(y_tr,pred_tr)
cm_te = cm(y_te,pred_te)

# Vis predictions
X_te_pc = pca.transform(X_te)
plt.subplot(121)
plt.scatter(X_te_pc[:,0],X_te_pc[:,1],c=y_te)
plt.subplot(122)
plt.scatter(X_te_pc[:,0],X_te_pc[:,1],c=pred_te)
plt.savefig("vis_test_pred.png")

print('Train cm ==> ')
print(cm_tr)
print('Test cm ==> ')
print(cm_te)

print("Train acc of ", cm_tr.diagonal().sum().astype(np.float32)/cm_tr.sum())
print("Test acc of ", score[1])
#print("Test f1 measure of ", f1(y_te,pred_te))
print("finished.")
