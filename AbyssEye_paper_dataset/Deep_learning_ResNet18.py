
######################################################################
#					deep learning ResNet-18 classifier               #
#                             Main Routine                           #
#                                                                    #
#      Uesed data set       :   trainin data and test data           #
#      Uesed classification :   ResNet18(Version 2)                  #
######################################################################

import keras
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import datetime
import os

from keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, Add 
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers 	import Adam
from tensorflow.keras.preprocessing.image	import ImageDataGenerator
from sklearn.model_selection 	import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils as np_utils

Load_param = False		
Save_param = True		
Only_eval  = False		
Plot_graph = True		
Plot_model = False		
Data_gener = True		
epochs     = 300		
batch_size = 64			
category   = 4			# classes
Model_name = "MyResNet18_model"
Model_name_2 = "MyResNet18_model_best"
gpu_id=0
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)

CUDA_VISIBLE_DEVICES=0

#---------------------
early_stopping =  EarlyStopping(
							monitor='val_loss',
							min_delta=0.0,
							patience=10)
modelCheckpoint = ModelCheckpoint(filepath = Model_name_2,
								  monitor='val_loss',
								  verbose=1,
								  save_best_only=True,
								  save_weights_only=False,
								  mode='min')

def set_seed(seed=151):
	tf.random.set_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	os.environ["PYTHONASHSEED"] = str(seed)
	tf.random.set_seed(0)
	tf.random.uniform([1])
	tf.random.uniform([1])
	tf.random.uniform([1])
	tf.random.uniform([1])

######################################################################
#          Visualization and graphing of learning history         #
######################################################################
def plot_history(history, epochs):
	

	fig = plt.figure(figsize=(10,4))
	ax1 = fig.add_subplot(1,2,1)
	ax2 = fig.add_subplot(1,2,2)
	ax1.plot(history['loss'],    label="for training")
	ax1.plot(history['val_loss'],label="for validation")
	ax1.set_title("model_loss")
	ax1.set_xlabel('epochs')
	ax1.legend(loc='upper right')

	ax2.plot(history['accuracy'],     label="for training")
	ax2.plot(history['val_accuracy'], label="for validation")
	ax2.set_title("model_accuracy")
	ax2.set_xlabel('epochs')
	ax2.legend(loc='lower right')
	plt.show()

	today  = datetime.datetime.now()
	f_name = 'fit_history' +str(epochs)+today.strftime('_%Y_%m_%d')+'.png'
	fig.savefig(f_name)
	plt.close()
# End of plot_history()

def load_network_model(Model_name_2):  
	print(Model_name_2,'load')
	my_model = keras.models.load_model(Model_name_2) 
	return my_model


def save_network_model(model, model_name):
	print(f'Save network information in the {model_name} folder.')
	model.save(model_name)
	return model

def evaluate_class(model, x_test, y_test):
	scores  = model.evaluate(x_test, y_test, verbose=1)
	metrics = model.metrics_names
	return metrics, scores

######################################################################
#                         ResNet-18 model                        #
######################################################################
def resnet(num_blocks, wide, img_size):
	input = Input(shape = img_size, dtype=tf.float32)
	num_filters = 64
	X = input
	X = Conv2D(num_filters,(3,3), padding='same', 
				kernel_initializer='he_normal')(X)
	short_cut = X
	X = BatchNormalization()(X)

	for i,blocks in enumerate(num_blocks):
		for j in range(blocks):
			if i > 0 and j == 0:
				short_cut=Conv2D(num_filters,(1,1), strides=(2,2),
							kernel_initializer='he_normal')(short_cut)
				X = Activation('relu')(X)
				X = Conv2D(num_filters,(3,3), strides=(2,2),padding='same',
							kernel_initializer='he_normal')(X)
				X = BatchNormalization()(X)
			else:
				X = Activation('relu')(X)
				X = Conv2D(num_filters,(3,3), padding='same',
							kernel_initializer='he_normal')(X)
				X = BatchNormalization()(X)

			X = Activation('relu')(X)
			X = Conv2D(num_filters,(3,3), padding='same',
					kernel_initializer='he_normal')(X)
			X = Add()([X, short_cut])
			short_cut = X
			X = BatchNormalization()(X)
		#End of FOR(j)
		num_filters *= wide
	#End of FOR(i)

	#Activation
	X = Activation('relu')(X)
	X = GlobalAveragePooling2D()(X)
	X = Dropout(0.4)(X)
	y = Dense(category, activation='softmax')(X)

	#model cliation
	model = Model(inputs=[input], outputs=[y])
	return model	#return model
#resnet()

######################################################################
#                           Scheduler                                # 
######################################################################
def step_decay(epoch):
	x = 0.001
	if epoch >= 100: x = 0.0005
	if epoch >= 150: x = 0.0001
	return x

from keras.callbacks import LearningRateScheduler
decay = LearningRateScheduler(step_decay, verbose=1)




##############################################################################
#						 get_random_eraser()                                 #
# https://github.com/yu4u/cutout-random-erasing/blob/master/random_eraser.py #
def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, 
						pixel_level=False):
	def eraser(input_img):
		if input_img.ndim == 3:
			img_h, img_w, img_c = input_img.shape
		elif input_img.ndim == 2:
			img_h, img_w = input_img.shape
		p_1 = np.random.rand()
		if p_1 > p:
			return input_img

		while True:
			s = np.random.uniform(s_l, s_h) * img_h * img_w
			r = np.random.uniform(r_1, r_2)
			w = int(np.sqrt(s / r))
			h = int(np.sqrt(s * r))
			left = np.random.randint(0, img_w)
			top = np.random.randint(0, img_h)
			if left + w <= img_w and top + h <= img_h:
				break

		if pixel_level:
			if input_img.ndim == 3:
				c = np.random.uniform(v_l, v_h, (h, w, img_c))
			if input_img.ndim == 2:
				c = np.random.uniform(v_l, v_h, (h, w))
		else:
			c = np.random.uniform(v_l, v_h)

		input_img[top:top + h, left:left + w] = c
		return input_img
	return eraser

######################## data augmentation ###############################
datagen = ImageDataGenerator(
		featurewise_center = False,	#set input mean to 0 over the dataset
		samplewise_center = False,	#set each sample mean to 0
		featurewise_std_normalization = False,#divide inputs by std of the dataset
		samplewise_std_normalization = False, #divide each input by its std
		zca_whitening = False,		#apply ZCA whitening
		zca_epsilon = 1e-06,		#epsilon for ZCA whitening
		rotation_range = 180,		#randomly rotate images from 0 to 180 degree
		width_shift_range = 0.5,	#randomly shift images horizontally
		height_shift_range = 0.5,	#randomly shift images vertically 
		shear_range = 0.0,			#set range for random shear
		zoom_range = 0.0,			#set range for random zoom
		channel_shift_range = 0,	#set range for random channel shifts
		fill_mode = 'nearest',		#set mode for filling points outside
		cval = 0.,					#value for fill_mode="constant"
		horizontal_flip = True,		#randomly flip images
		vertical_flip = True,		#randomly flip images
		rescale = None,				#set rescaling factor

		#set function that will be applied on each input
		preprocessing_function = get_random_eraser(p=0.5,s_l=0.02,
					s_h=0.1,r_1=0.3,r_2=1/0.3,v_l=0,v_h=0),
		validation_split = 0  
		)



#######################################################################
# Writing and reading of training parameters, flags for model drawing #
#######################################################################
def main(epochs=10):
	print('******** Initial Conditions ********')
	print('Load_param:',   Load_param,end='')
	print('\tSave_param:', Save_param)
	print('Data_gener:',   Data_gener,end='')
	print('\tPlot_graph:', Plot_graph)

######################## Training dataset #########################
	X_train = np.load('../TrainData/X_train.npy') 
	Y_train = np.load('../TrainData/Y_train_new.npy') 
	print(f'\n---- Total training data :{X_train.shape[0]} ----\n\n')

#################   train data shuffle  #################

	order = list(range(X_train.shape[0]))
	random.shuffle(order)
	X_train = X_train[order]
	Y_train = Y_train[order]

	X_test  = np.load('../TestData/X_test.npy')
	Y_test  = np.load('../TestData/Y_test.npy')
	Y_test0 = Y_test.copy()
	print(f'\n---- Total test data:{X_test.shape[0]} ----\n\n')
	X_train = X_train / 255.0
	X_test  = X_test  / 255.0	
	print(X_test[0].shape)

	print('\n\t\t******* The labels of all test data *******')
	for i in range(len(Y_test)):
		if i%10 == 0: print()
		print(f' ({i:3d},{Y_test[i]})',end='')

	count = np.zeros((category,), dtype = np.int16)
	for i in range(Y_train.shape[0]):
		count[Y_train[i]] += 1
	print('\n\n\t*** Train Label Distribution:',count )
	
	count = np.zeros((category,), dtype = np.int16)
	for i in range(Y_test.shape[0]):
		count[Y_test[i]] += 1
	print('\n\n\t*** Test Label Distribution:',count )

	input_img_shape = X_train.shape[1:4]

	############# Convert label data to categorical data  ##############
	Y_train = np_utils.to_categorical(Y_train, num_classes=category)
	Y_test  = np_utils.to_categorical(Y_test,  num_classes=category)

	###################################################################
	#                    Validation data (0.1-0.25 )                 #
	###################################################################
	(X_train,X_valid,Y_train,Y_valid) = train_test_split(X_train, Y_train, test_size=0.2)

	
	################  Creation of ResNet model instances ##################

	model = resnet(num_blocks =[2,2,2,2], wide=2, img_size=input_img_shape)
	model.summary()

	################# Learning Model Graphs ################
	if Plot_model == True:
		from keras.utils import plot_model
		plot_model(model, to_file='ResNet18_model.png', expand_nested=False)

	#################### Read in the learned weight coefficients #####################
	if Load_param == True:	
		New_model = load_network_model(Model_name_2)

	model.compile(Adam(lr=1e-3), loss="categorical_crossentropy",
			metrics=["accuracy"])
	
	############ Execute only the classification process for unlearned patterns ###############
	if Only_eval == True:
		score = evaluate_class(New_model, X_test, Y_test)
		#score = model.evaluate(New_model,X_test,Y_test, verbose=1)
		print('loss=', score[0])
		print('accuracy=', score[1])

	else:
	################## Update learning based on data expansion ###################
		class_weight ={ 0:1.0, 1:1.0, 2:1.0, 3:1.0}
		
		if Data_gener == True:
			result = model.fit_generator(
				datagen.flow(X_train, Y_train, batch_size=batch_size),
				steps_per_epoch=len(X_train)//batch_size, epochs=epochs,
				class_weight=class_weight,
				validation_data=(X_valid,Y_valid),
				callbacks=[ modelCheckpoint, early_stopping])
		else:
			print('Y_train.shape:',Y_train.shape)
			print('X_train.shapr:',X_train.shape)
			result = model.fit(X_train, Y_train, batch_size=batch_size, 
				epochs=epochs,
				validation_data=(X_valid,Y_valid),
				class_weight=class_weight,verbose=1,
				callbacks=[ modelCheckpoint, early_stopping])

		########### Creating frames for drawing learning history graphs ############
		if Plot_graph == True:
			plot_history(result.history, epochs)


		#################### Store learned weight coefficients ###################
		if Save_param == True:
			save_network_model(model,Model_name)

		################### Unlearned pattern classification process ####################
		New_model = load_network_model(Model_name_2)
		score = evaluate_class(New_model, X_test, Y_test)
		print('loss=', score[0])
		print('accuracy=', score[1])

		######################## Display of misclassified images #########################
		predicts = model.predict(X_test)         
		check    = np.zeros(predicts.shape[0], dtype = np.uint8)
		fails    = np.zeros(predicts.shape[0], dtype = np.uint8)
		est = np.argmax(predicts,axis=1)
		Y_test0  = Y_test0.reshape(Y_test0.shape[0],)

		for i in range(predicts.shape[0]):
			if predicts[i,0] < predicts[i,1]: check[i] = 1
		count = 0
		for i in range(predicts.shape[0]):
			if Y_test0[i] == 0: count += 1
		print(f'\n\n\t*** Total number of positive data: {count} *** \n\n')
		est_2 = np.argmax(Y_test,axis=1)
		for cl in range(4):
			TP = 0
			FN = 0
			FP = 0
			TN = 0
			for i in range(len(est)):
				if est[i] == cl and est_2[i] == cl: TP += 1
				if est[i] != cl and est_2[i] == cl: FN += 1
				if est[i] == cl and est_2[i] != cl: FP += 1
				if est[i] != cl and est_2[i] != cl: TN += 1 
			Recall = TP/(TP+FN)
			Precision = TP/(TP+FP)
			F1 = (2*Recall*Precision)/(Recall+Precision)
			print("class:"+str(cl))
			print(Recall)
			print(Precision)
			print(F1)
	#############################################################################
	# Detailed display of misclassification (serial numbers and image patterns) #
	#############################################################################
		est = np.argmax(predicts,axis=1)
		estimates = np.argmax(predicts,axis=1)
		count=0
		for i in range(estimates.shape[0]):
			if estimates[i] != Y_test0[i]:
				if count%10 == 0:   
					print(f'({i:4d},{Y_test0[i]},{estimates[i]}) ',end='')
				count+=1

		confusion_matrix = np.zeros((category,category))
		for i in range(estimates.shape[0]):
				confusion_matrix[Y_test0[i],estimates[i]] += 1
		
	#### Accuracy evaluation with the same number of sheets in each class ####
		
		XX_test = np.empty((0,input_img_shape[0],input_img_shape[1],input_img_shape[2]))
		YY_test = np.empty(0)
		count0=0
		count1=0
		count2=0
		count3=0
		for i in range(X_test.shape[0]):
			if Y_test0[i] == 0 and count0 <450:
				XX_test = np.append(XX_test,X_test[i][np.newaxis],axis=0)
				YY_test = np.append(YY_test,Y_test0[i])
				count0+=1
			elif Y_test0[i] == 1 and count1 < 450:
				XX_test = np.append(XX_test,X_test[i][np.newaxis],axis=0)
				YY_test = np.append(YY_test,Y_test0[i])
				count1+=1
			elif Y_test0[i] == 2 and count2 < 450:
				XX_test = np.append(XX_test,X_test[i][np.newaxis],axis=0)
				YY_test = np.append(YY_test,Y_test0[i])
				count2+=1
			elif Y_test0[i] == 3 and count3 < 450:
				XX_test = np.append(XX_test,X_test[i][np.newaxis],axis=0)
				YY_test = np.append(YY_test,Y_test0[i])
				count3+=1
			elif count0 < 450 or count1 < 450 or count2 < 450 or count3 < 450:
				continue
			else:
				print(f"{count0},{count1},{count2},{count3}")
				break        
		YY_test  = keras.utils.to_categorical(YY_test,  num_classes=category)
		score = model.evaluate(XX_test,YY_test,verbose=1)
		print('loss=', score[0])
		print('accuracy=', score[1])
	print(sum(estimates == 0))
	print(sum(estimates == 1))
	print(sum(estimates == 2))
	print(sum(estimates == 3))

# End of Main

######################################################################
if __name__ == '__main__' :
	main(epochs)


#category,input_img_shape
