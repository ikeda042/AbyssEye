######################################################################
## Training data integration system from individual training images ##
######################################################################
import os,sys
import numpy as np


"""

 >python data_integration.py


>>>import data_integration_1 as da
>>>holder_list =['../TrainData0/']
>>>da.main(holder_list)
>>>holder_list = ['../TestData1/']
>>>da.main(holder_list)

>>>holderList =['../TrainData1/', '../TestData1/']
>>>da.main(holderList)

"""

holderList =['../TrainData1/', '../TestData1/']

######################################################################
def main(holderList=holderList):
	#Organize by individual folder in the folder list
	for aHolder in holderList:
		print('aHolder:', aHolder)

		if aHolder.find('Train') > 0:
			fBody = 'train'
		elif aHolder.find('Test')> 0:
			fBody = 'test'
		else:
			sys.exit(0)

		#Get all file names in a folder
		allFiles = os.listdir(aHolder)
		npys   = [file for file in allFiles if file.find('.npy') > 0]
		x_npys = [i for i in npys if i.find('_x_train') > -1]
		y_npys = [i for i in npys if i.find('_y_train') > -1]
		print('x_npys:', x_npys)
		print('y_npys:', y_npys) 

		for i in range(len(x_npys)):
			data = np.load(aHolder + x_npys[i])
			print(x_npys[i])
			if i == 0:
				x_conc = data
			else:
				x_conc = np.vstack((x_conc, data))

		###### "X_train.npy(X_test.npy)" #####
		np.save(aHolder+ 'X_'+ fBody + '.npy', x_conc)
		#Integrated Data Name ：'X_????.npy'
		print('\n')

		for i in range(len(y_npys)):
			data = np.load(aHolder + y_npys[i])     #Import of ??_y_train.np
			length =data.shape[0]
			print(y_npys[i])
			data = data.reshape(length,1)
			if i == 0:
				y_conc = data
			else:
				y_conc = np.vstack((y_conc, data))

		#### "Y_train.npy(Y_test.npy)" ####
		np.save(aHolder+'Y_' + fBody + '.npy', y_conc)  
		#Integrated Data Name ：'Y_????.npy'

	#Process two folders		
	print('Completed')

#END of main()


if __name__ =='__main__' :
	main(holderList)

