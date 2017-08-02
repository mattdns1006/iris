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

class Iris_model():	
	def __init__(self,batch_size,learning_rate,test_size=0.2):
		data = load_iris()
		self.label_names = data.target_names.tolist()
		self.labels = data.target_names
		self.X = data.data
		self.Y = keras.utils.to_categorical(data.target)

		self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(self.X,self.Y,test_size=test_size,random_state=1006)
		self.bs = batch_size
		self.lr = learning_rate
		self.compile()
	
		# Quick vis of train test distrbution for y
		bins = 3
		plt.hist(self.y_tr.argmax(1).flatten(),bins=3,ec='black',alpha=0.5,label="Train distribution")
		plt.hist(self.y_te.argmax(1).flatten(),bins=3,ec='black',alpha=0.5,label="Test distribution")
		plt.legend()
		plt.savefig("tr_te_dist.png")
		plt.clf()

	def load_data(self,train_test):
		if train_test == "train":
			X,Y = self.X_tr, self.y_tr 
		else:
			X,Y = self.X_te, self.y_te
		frm = 0
		to = frm + self.bs
		n = X.shape[0]
		while True:
			if frm >= n:
				frm = 0 
				to = frm + self.bs 
			x = X[frm:to]
			y = Y[frm:to]
			frm = min(frm+self.bs,n)
			to = min(to+self.bs,n)
			yield (x, y)

	def train(self,epochs):
		n_train = self.X_tr.shape[0]
		steps_per_epoch = n_train/self.bs
		self.model.fit_generator(self.load_data("train"),
				    steps_per_epoch=steps_per_epoch,
				    nb_epoch = epochs,
				    verbose=2)
	
	def test(self):
		score = self.model.evaluate(self.X_te,self.y_te,
					batch_size=self.bs,
					verbose=1)

		print("*"*20)
		print("Test acc of ", score[1])
		self.pred_te = self.model.predict(self.X_te).argmax(1)
		y_te = self.y_te.argmax(1)
		cm_te = cm(y_te,self.pred_te)
		print('Test cm ==> ')
		print(cm_te)
		print("*"*20)

	def compile(self):
		model = Sequential()
		model.add(Dense(10,input_shape=(4,)))
		model.add(bn())
		model.add(Activation('relu'))
		model.add(Dense(10))
		model.add(bn())
		model.add(Activation('relu'))
		model.add(Dense(3))
		model.add(Activation('softmax'))
		adam = optimizers.Adam(lr=self.lr)
		model.compile(loss='categorical_crossentropy',
				optimizer=adam,
				metrics=['accuracy'])
		self.model = model
		plot_model(model,show_shapes=True,to_file='model.png',rankdir='TB')
		model.summary()

	def save(self,name='weights.h5'):
		self.model.save(name)
		print("Saved model in {0}".format(name))

	def pca_plot(self):

		# Pca XY plot
		pca = decomposition.PCA(n_components=2)
		
		# Train
		pca.fit(self.X_tr)
		X_tr_pc = pca.transform(self.X_tr)
		plt.scatter(X_tr_pc[:,0],X_tr_pc[:,1],c=self.y_tr.argmax(1))
		plt.savefig("train.png")
		plt.clf()

		# Test truth pred
		self.test() # to get predictions
		X_te_pc = pca.transform(self.X_te)
		plt.subplot(121)
		plt.scatter(X_te_pc[:,0],X_te_pc[:,1],c=self.y_te)
		plt.subplot(122)
		plt.scatter(X_te_pc[:,0],X_te_pc[:,1],c=self.pred_te)
		plt.savefig("vis_test_pred.png")
		plt.clf()


if __name__ == "__main__":
	batch_size = 5 
	epochs = 20
	lr = 0.01
	test_size = 0.2
	iris_model = Iris_model(batch_size,lr,test_size=test_size)	
	iris_model.train(epochs)
	#iris_model.test()
	iris_model.pca_plot()
	print("Finished.")
