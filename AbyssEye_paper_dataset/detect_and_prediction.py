###########################################################################################
# Detection of candidate microbial regions and category prediction by trained predictors  #
###########################################################################################

###################### Loading various modules #######################
import	os, glob, pathlib, sys, cv2
import keras
import	matplotlib.pyplot	as plt
import	numpy 				as np
import json
from	skimage.feature 	import peak_local_max
from scipy.stats import percentileofscore
######################## Global Variable Declaration ###########################
global img0, img1, mask, img2, msk2

############################## constant declaration ###############################
Height        = 48
Width         = 48
num_of_pixels = Height * Width
kernel_size1  = 3
kernel_size2  = 3
iterations1   = 1
iterations2   = 2
min_distance  = 0
green_rate    = 0.07

class_colors  = {0:'yellow',1:'lightgreen',2:'lightblue',3:'bisque'}
category      = 4
data_holder   = '../TestData/'
img_file_type = '.tif'
result_holder = '../Result/'
Model_name = 'MyResNet18_model_best'




################### loading externally stored networks ##################
def load_network_model(Model_name):
	print(Model_name,'load')
	my_model = keras.models.load_model(Model_name, compile=False)
	print(f'=== End of {Model_name} load ===')
	return my_model

########## Listing of files with a specific extension in a specified holder ###########
def make_file_list(directory,img_type):
	allFiles = os.listdir(directory)
	imgs = [file for file in allFiles if file.find(img_type) > 0]
	return imgs
#End of make_file_list() definition


############## Extract the rectangle information of a specific ROI in the ROI table ###############         
def get_fixed_roi_area(roi_info,no):
	roi = roi_info[no]
	xs, ys = roi["ST"]
	xe, ye = roi["EN"]
	return xs,ys,xe,ye
#End of get_fixed_roi_area()


#######################################################################
#                     morphological operation                         #
#######################################################################
def morphology(pic,op):
	kernel = np.ones((3,3), dtype=np.uint8)
	if op == 'dilate':
		return(cv2.dilate(pic, kernel, iterations=1))
	else:
		return(cv2.erode (pic, kernel, iterations=1))

##############################################################################################################
# Acquire luminance values corresponding to a certain amount of green highlight component from the G-channel #
##############################################################################################################
def find_thresh_from_percentile(green, green_rate):
	

	height, width = green.shape[0:2]
	num_of_pixels = height * width
	hist  = np.histogram(green, bins=256, range=(0,256))
	cumm  = np.cumsum(hist[0])   
	p_pos = percentileofscore(cumm,(1-green_rate)*num_of_pixels, kind='strict')
	p_pos = int(p_pos *255 *0.01)	
	print('\n cutoff tone value:',p_pos)
	return p_pos
#End of find_thresh_from_percentile() 


######################################################################
#         Pick up ROI candidate areas by color conditions            #
######################################################################
def extract_rois(green_rate):
	global mask

	red   = img1[:,:,0]
	green = img1[:,:,1]
	blue  = img1[:,:,2]
	cutoff = np.zeros(green.shape[0:2], np.uint8)

	cutoff[:,:] = green[:,:]		
	thresh = find_thresh_from_percentile(green, green_rate)

	################ Setting conditions for candidate area extraction ####################
	mask1 = np.logical_and(
				np.logical_and(green > thresh, green > 30),
				green/red > 1.0)
	mask2 = np.logical_and(
				np.logical_and(green < thresh, green > 30),
				green/red >= 1.5), 
	mask0 = (mask1 | mask2).reshape(mask1.shape)

	cutoff[np.logical_not(mask0)] = 0	

	kernel3  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) 

	tmp1  = cv2.morphologyEx(cutoff,cv2.MORPH_DILATE,kernel3,iterations=2)

	coord1 = peak_local_max(tmp1, min_distance=min_distance)
	num_of_maxima = coord1.shape[0]
	tmp1[:,:] = 0
	for i in range(num_of_maxima):
		y = int(coord1[i][0])
		x = int(coord1[i][1])
		tmp1[y,x] = 1

	cutoff   = cv2.morphologyEx(cutoff,cv2.MORPH_DILATE,kernel3,iterations=1)
	coord2   = peak_local_max(cutoff, min_distance=0)

	for i in range(coord2.shape[0]):
		mask[int(coord2[i][0]),int(coord2[i][1])] = 255

	################ BoundingBox calculations for new areas after integration  ##################
	nlabels,centers = cv2.connectedComponentsWithStats(tmp1)

	roi_info = []   
	ht, wh = img1.shape[0:2]
	for i in range(1,nlabels):
		xc, yc = int(centers[i][0]), int(centers[i][1])
		ys, xs = yc-int(Height/2),   xc-int(Width/2)
		ye, xe = yc+int(Height/2),   xc+int(Width/2)	
		if ys < 0:
			ys, ye = 0, Height
		elif ye > ht:
			ys, ye = ht-Height, ht
		if xs < 0 :
			xs, xe = 0, Width
		elif xe > wh:
			xs, xe = wh-Width, wh

		yc, xc = int(ys+ye)/2, int(xs+xe)/2
		roi = { "ID" : i,				
				"ST" : [xs, ys], "EN" : [xe, ye],
				"CE" : [xc, yc], "CL" : 0,
		}
		roi_info.append(roi)	

	return roi_info
#End of extract_rois() definition

########################################################################################################
## Processing of integrating, exporting, and displaying ROI candidate areas into a 4-dimensional array #
########################################################################################################
def push_roi_areas(roi_info):
    num_of_rois = len(roi_info)
    img_stack   = np.zeros((num_of_rois,Height,Width,3), dtype=np.uint8)
    msk_stack   = np.zeros((num_of_rois,Height,Width,1), dtype=np.uint8)

    for i in range(num_of_rois):
        xs,ys,xe,ye = get_fixed_roi_area(roi_info, i)
        img = img0[ys:ye,xs:xe,:]
        msk = mask[ys:ye,xs:xe]

        img_stack[i] = img
        msk_stack[i] = np.expand_dims(msk, -1)

    return img_stack, msk_stack
#End of push_roi_areas() definition

################# write_ROIs_in_stacks() #######################
def write_rois_in_stacks(t_path,result_dir,area_stack, mask_stack,rois):
	area_name = t_path.stem + '_x_pred.npy'
	mask_name = t_path.stem + '_x_mask.npy'
	labl_name = t_path.stem + '_y_pred.npy'
	area_path = os.path.join(result_dir,area_name)
	mask_path = os.path.join(result_dir,mask_name)
	labl_path = os.path.join(result_dir,labl_name)

	np.save(area_path, area_stack)
	np.save(mask_path, mask_stack)

	labels = []		
	for i in range(len(rois)):
		labels.append(rois[i]["CL"])

	labels = np.array(labels)
	np.save(labl_path, labels)
#End of write_rois_in_stacks() definition

#######################################################################
#                      write_ROIs_in_json()                           #
#######################################################################
def write_rois_in_json(t_path, result_dir, roi_info):
	
	json_file = t_path.stem + '.json'
	json_path = os.path.join(result_dir, json_file)

	with open(json_path, mode='w') as fp:
		json.dump(roi_info, fp)
		print('JSON File output')

# End of write_rois_in_json() definition
		
def evaluate_class(model, x_test, y_test):
	scores  = model.evaluate(x_test, y_test, verbose=1)
	metrics = model.metrics_names
	return metrics, scores

###########################################################################
###                       visualize_predictions()                       ###
###########################################################################
def visualize_predictions(images, predicts, title, col=10, row=8) :

	no = images.shape[0]	
	flag     = True
	g_width  = 10					
	g_height = 10					
	square   = min(g_width/col, g_height/row)
	g_width  = square * col; g_height = square * row 	
	index    = 0

	while flag :
		fig = plt.figure(figsize=(g_width, g_height))	
		fig.suptitle(f'{title}: ({index})')
		for i in range(row):
			if flag == False: 
				break
			for j in range(col):
				ax = fig.add_subplot(row, col,i*col+j+1)
				ax.imshow(images[index] )
				ax.text(0.5,6.0,str( int(np.argmax(predicts[index]))),size=8, color="white")
				ax.text(0.5,47.0,str(round(np.max(predicts[index]),2)),size=8, color="white")
				ax.text(30.0,47.0,str(round(np.max(predicts[index]) - sorted(predicts[index])[-2],2)),size=8, color="white")
				index += 1
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)
				if index >= no :
					flag = False
					break
		plt.show()

#######################################################################
#                            show_ROIs()                              #
#######################################################################

def show_ROIs(images, masks, total, title, col=10, row=8) :
	
	flag     = True
	g_width  = 10					
	g_height = 12					
	square   = min(g_width/col, g_height/row)
	g_width  = square * col; g_height = square * row 	

	index0,index1 = 0, 0
	while flag :
		fig = plt.figure(figsize=(g_width, g_height))	
		fig.suptitle(f'{title}: ({index0})')

		for i in range(0, row, 2):
			if flag == False: 
				break
			index1 = 0
			for j in range(col):
				ax = fig.add_subplot(row, col,i*col+j+1)	
				ax.imshow(images[index0+index1])			
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)
				ax = fig.add_subplot(row,col,(i+1)*col+j+1)
				ax.imshow(masks[index0+index1].reshape(Height,Width), cmap='gray')
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)
				index1 += 1
				if index0+index1 >= total :
					flag = False
					break
			index0 += col
		plt.show()

#End of show_ROIs() definition



#######################################################################
#                              main()                               #
#######################################################################
def main( ):
	global img0, img1, mask, img2, msk2
	model = load_network_model(Model_name)
        
	########## Extraction process of candidate microorganism regions from unlearned images #########
	img_files = make_file_list(data_holder,img_file_type)
	print(img_files)

	img_stack2 = np.empty((0,48,48,3))
	
	CL = np.empty((0))
	for a_file in img_files:		
		img_path = os.path.join(data_holder, a_file)
		
		img   = cv2.imread(img_path)	
		if img is None:
			print("Failed to load image. Please check the file name.")
			sys.exit(1)

		img   = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		img0  = cv2.resize(img,(round(img.shape[1]/2),round(img.shape[0]/2)))
		img1  = img0.copy()
		mask  = np.zeros(img0.shape[0:2], np.uint8)
		img2  = np.zeros((Height,Width,3), dtype=np.uint8)
		msk2  = np.zeros((Height,Width),   dtype=np.uint8)

		####################### Generation of ROI information ########################
		roi_info = extract_rois(green_rate) 
		num_of_rois = len(roi_info)
		print("num_of_rois:", num_of_rois)

		#################### Export process of ROI information #####################
		temp_path = os.path.join(result_holder, a_file)
		temp_path = pathlib.Path(temp_path)
		img_stack, msk_stack = push_roi_areas(roi_info) 
		img_stack = img_stack / 255.0
		img_stack2 = np.concatenate([img_stack2,img_stack],0)
		print(img_stack.shape)
		
		predicts = model.predict(img_stack)   
		visualize_predictions(img_stack,predicts,"Predicted ROI and Label:")
		print(predicts)
	
		for i in range(num_of_rois):          
			roi_info[i]["CL"] = int(np.argmax(predicts[i]))
			CL = np.append(CL,roi_info[i]["CL"])	

		write_rois_in_stacks(temp_path,result_holder,img_stack,msk_stack,roi_info)
		write_rois_in_json(temp_path, result_holder, roi_info)
		
			
	#End of FOR(a_file) loop
#End of main() definition

if __name__ == '__main__':
	main( )

