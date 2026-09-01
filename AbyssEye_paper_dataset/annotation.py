#######################################################################
#        Annotation system with graphical user interface              #
#######################################################################


###################### Loading various modules #######################
import	os, glob, pathlib, sys, cv2
import	matplotlib.pyplot	as plt
import	numpy 				as np
from 	matplotlib.widgets 	import Button, RadioButtons
from	skimage.feature 	import peak_local_max
from scipy.stats import percentileofscore
import cv2
######################## Global Variable Declaration ###########################
global fig,ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9
global button0, button3, button4,button6, button7, button8
global img0, img1, mask, img2, msk2

Height        = 48
Width         = 48
kernel_size1  = 3
kernel_size2  = 3
iterations1   = 1
iterations2   = 2
min_distance  = 0
green_rate    = 0.07
class_colors  = {0:'yellow',1:'lightgreen',2:'lightblue',3:'bisque', 4:'yellow'}
dataset_dir = "../Photo/"

#######################################################################
def make_file_list(directory):
	allFiles = os.listdir(directory)

	bmp_imgs = [file for file in allFiles if file.find('.tif') > 0]
	return bmp_imgs
#End of make_file_list()


######################## Declaration of working image array #########################

############## ROI_DecisionManager( ) #################
class ROI_DecisionManager( ):
	def __init__(self, color_assignment):
		self.current  = 0
		self.roi_info = []
		self.length   = 0
		self.cls_data = {'class0':0,'class1':1,'class2':2,'class3':3}
		self.col_data = color_assignment
	#End of __init__()

	def decide_roi_class(self,label):
		global mask, msk2
		cl_no = self.cls_data[label]
		roi = self.roi_info[self.current]
		roi["CL"] = cl_no
		xs,ys,xe,ye = get_fixed_roi_area(self.roi_info,self.current)
		mask[ys:ye, xs:xe] = msk2[:,:]
		color = self.col_data[cl_no]
		ax9.set_facecolor(color)
		fig.canvas.draw_idle()

	def increment_index(self):		
		global img2,msk2
		roi_info = self.roi_info
		self.current = (self.current+1)%self.length
		roi = self.roi_info[self.current]
		create_roi_image(roi_info, self.current)
		ax2.imshow(img2)
		ax5.imshow(msk2, cmap='gray')
		del plt.gca().texts[-1]			
		strng = "< " + str(self.current+1)+ "/ " + str(self.length) + ">"
		ax9.text(0.09, 0.30, strng,fontsize=14)
		color = self.col_data[roi["CL"]]
		ax9.set_facecolor(color)
		fig.canvas.draw_idle()

	def decrement_index(self):		
		global img2, msk2
		roi_info = self.roi_info
		self.current = (self.current-1)%self.length
		roi = self.roi_info[self.current]
		create_roi_image(roi_info, self.current)
		ax2.imshow(img2)
		ax5.imshow(msk2,cmap='gray')
		del plt.gca().texts[-1]			
		strng = "< " + str(self.current+1) + "/ " + str(self.length) + ">"
		ax9.text(0.09, 0.30, strng, fontsize=14)
		color = self.col_data[roi["CL"]]
		ax9.set_facecolor(color)
		fig.canvas.draw_idle()

	def terminate_gui(self):
		plt.close()
		print('GUI Termination')

	def erode_mask(self):
		global msk2
		tmp = morphology(msk2, op = 'erode')
		for i in range(Height):
			for j in range(Width):
				msk2[i,j] = tmp[i,j]
		ax5.imshow(msk2, cmap='gray')
		fig.canvas.draw_idle()

	def dilate_mask(self):
		global msk2
		tmp = morphology(msk2, op='dilate')
		for i in range(Height):
			for j in range(Width):
				msk2[i,j] =tmp[i,j]
		ax5.imshow(msk2,cmap='gray')
		fig.canvas.draw_idle()
		
#END of ROI_DecisionManager


#######################################################################
#          Fixed area cutout around a candidate point                 #
#######################################################################
def get_fixed_roi_area(roi_info,no):
	roi = roi_info[no]
	xs, ys = roi["ST"]
	xe, ye = roi["EN"]
	return xs,ys,xe,ye

################### Display the area of the ROI of interest on the AX1 image####################

def mark_roi_areas(rman,roi_info,prev,current):
	global img0,img1
	if prev == current:
		xs, ys, xe, ye = get_fixed_roi_area(roi_info, current)
		img1[ys:ye,xs:xe,2] = 255		
	else:
		xs, ys, xe, ye = get_fixed_roi_area(roi_info, prev)
		img1[ys:ye,xs:xe,2] = img0[ys:ye,xs:xe,2]
		xs, ys, xe, ye = get_fixed_roi_area(roi_info, current)
		img1[ys:ye,xs:xe,2] = 255		

#######################################################################
##  Creation of images to be displayed on AX2 and AX5                ##
#######################################################################
def create_roi_image(roi_info, n):
	global img0,img2, msk2
	xs,ys,xe,ye = get_fixed_roi_area(roi_info, n)
	img2[:,:,:] = img0[ys:ye,xs:xe,:]
	msk2[:,:]   = mask[ys:ye,xs:xe].copy()

#######################################################################
#                  morphological operation                            #
#######################################################################
def morphology(pic,op):
	kernel = np.ones((3,3), dtype=np.uint8)
	if op == 'dilate':
		return(cv2.dilate(pic, kernel, iterations=1))
	else:
		return(cv2.erode (pic, kernel, iterations=1))

###############################################################################################################
# Acquire luminance values corresponding to a certain amount of green highlight component from the G-channel  #
###############################################################################################################
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
#            Pick up ROI candidate areas by color conditions         #
######################################################################
def extract_rois(roi_manager, green_rate):
	global mask

	red   = img1[:,:,0]
	green = img1[:,:,1]
	blue  = img1[:,:,2]
	cutoff = np.zeros(green.shape[0:2], np.uint8)

	cutoff[:,:] = green[:,:]		
	thresh = find_thresh_from_percentile(green, green_rate)

	############## Setting conditions for candidate area extraction ##############

	mask1 = np.logical_and(
				np.logical_and(green > thresh, green > 30),
				green/red > 1.0) 
	mask2 = np.logical_and(
				np.logical_and(green < thresh, green > 30),
				green/red >= 1.5),
	mask0 = (mask1 | mask2).reshape(mask1.shape)

	cutoff[np.logical_not(mask0)] = 0	
	cv2.imwrite('cutoff.tif',cutoff)
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

	################ BoundingBox calculations for new areas after integration ##################
	nlabels,label,stats,centers = cv2.connectedComponentsWithStats(tmp1)
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

	roi_manager.roi_info = roi_info
	roi_manager.length   = len(roi_info)
	return (roi_info, nlabels-1)
#End of pickup_rois()

	########################################################################################################
	## Processing of integrating, exporting, and displaying ROI candidate areas into a 4-dimensional array #
	########################################################################################################
def push_roi_areas(roi_info):
	num_of_rois = len(roi_info)
	img_stack = []
	msk_stack = []

	for i in range(num_of_rois):
		xs,ys,xe,ye = get_fixed_roi_area(roi_info, i)
		img = img0[ys:ye,xs:xe,:]
		msk = mask[ys:ye,xs:xe]
		
		if len(img_stack) == 0:
			img_stack = img.reshape(1,Height,Width,3)
			msk_stack = msk.reshape(1,Height,Width,1)
		else:
			img_stack = np.vstack((img_stack,img.reshape(1,Height,Width,3)))
			msk_stack = np.vstack((msk_stack,msk.reshape(1,Height,Width,1)))

	return img_stack, msk_stack
#End of push_roi_areas()

################# write_ROIs_in_stacks() #######################
def write_rois_in_stacks(img_path,dataset_dir,img_stack, msk_stack,rois):
	img_name = img_path.stem + '_x_train.npy'
	msk_name = img_path.stem + '_msk.npy'
	lbl_name = img_path.stem + '_y_train.npy'
	img_path = os.path.join(dataset_dir,img_name)
	msk_path = os.path.join(dataset_dir,msk_name)
	lbl_path = os.path.join(dataset_dir,lbl_name)
	np.save(img_path, img_stack)
	np.save(msk_path, msk_stack)
	labels = []		
	for i in range(len(rois)):
		labels.append(rois[i]["CL"])

	labels = np.array(labels)
	np.save(lbl_path, labels)
#End of write_rois_in_stacks

#######################################################################
#                     write_ROIs_in_json()                            #
#######################################################################
def write_rois_in_json(img_path, dataset_dir, final_roi):
	import json
	json_file = img_path.stem + '.json'
	json_path = os.path.join(dataset_dir, json_file)
	fp = open(json_path, 'w')
	json.dump(final_roi, fp)
	print('JSON File output')
	fp.close()
# End of write_rois_in_json()

#######################################################################
#                            show_ROIs()                              #
#######################################################################
def show_ROIs(images, masks, total, title, col=10, row=8) :
	# images:	Integrated ROI image data
	# masks :   Integrated mask image data
	# total :	Total number of displayed images
	# title :   Title of displayed image
	# col   :	Number of images displayed per line (default value: 10)
	# row   :	Number of rows for displaying images (default value: 8)
	
	flag     = True
	g_width  = 10					
	g_height = 12					
	square   = min(g_width/col, g_height/row)
	g_width  = square * col; g_height = square * row 	

	index0,index1 = 0, 0
	while flag :
		fig = plt.figure(figsize=(g_width, g_height))	
		fig.suptitle(f'Candidates: {title}({index0})')

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
#END of show_ROIs()


#######################################################################
#                          gui_setup()                                #
#######################################################################
def gui_setup():
	global fig,ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9
	global button0,button3,button4,button6,button7,button8

	################# matplotlib drawing instance creation ##################
	fig = plt.figure(figsize=(16,12))
	ax0 = fig.add_axes([0.42, 0.10, 0.10,0.14],title='Register')#register Button window
	ax1 = fig.add_axes([0.10, 0.29, 0.80,0.70])		#original image display window
	ax2 = fig.add_axes([0.10, 0.07, 0.10,0.13])		#ROI image display window
	ax3 = fig.add_axes([0.11, 0.22, 0.04,0.04])		#backward Button1 window
	ax4 = fig.add_axes([0.16, 0.22, 0.04,0.04])		#forward Button2 window
	ax5 = fig.add_axes([0.22, 0.07, 0.10,0.13])		#Binarized image display window
	ax6 = fig.add_axes([0.56, 0.08, 0.06,0.04])		#exit button window
	ax7 = fig.add_axes([0.33, 0.08, 0.04,0.04])		#erode button window
	ax8 = fig.add_axes([0.33, 0.16, 0.04,0.04])		#ROI number display window
	ax9 = fig.add_axes([0.23, 0.22, 0.08,0.04])		#ROI number display window

	################## Callback settings for drawing-related buttons ###############
	button0 = RadioButtons(ax0, ['class0','class1','class2','class3'],active=0,
					   activecolor='black')
	button6 = Button(ax6, 'Terminate')
	button3 = Button(ax3, 'Backward')
	button4 = Button(ax4, 'Forward')
	button7 = Button(ax7, 'Erode')
	button8 = Button(ax8, 'Dilate')
#End of gui_setup
#######################################################################



#######################################################################
#                              main()                                 # 
#######################################################################
def main( ):
	global fig,ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9
	global button0, button3, button4,button6, button7, button8
	global img0, img1, mask, img2, msk2

	img_files = make_file_list(dataset_dir)
	print(img_files)

	img_path = os.path.join(dataset_dir, img_files[0])
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

	gui_setup()

	############### ROI management object instance creation ###############
	roi_manager = ROI_DecisionManager(class_colors)

	roi_info, num_of_rois = extract_rois(roi_manager,green_rate)

	######################### Generate initial screen ##########################
	for i in range(num_of_rois):
		mark_roi_areas(roi_manager,roi_info, i, i)
	ax1.imshow(img1)				#Display of all candidate areas
	create_roi_image(roi_info, 0)	#ROI cutout of candidate region 0
	ax2.imshow(img2)				#Display enlarged ROI area on ax2
	ax5.imshow(msk2,cmap='gray')	#Mask display of ROI area on ax5
	ax9.set_xticks([])				#Suppresses ax9 coordinate axis display
	ax9.set_yticks([])				#Suppresses ax9 coordinate axis display

	strng = "< " + str(1) + "/ " + str(len(roi_info)) + ">"
	ax9.text(0.1, 0.30, strng, fontsize=14) 	
	ax9.set_facecolor(class_colors[0])			

	########### Setup callback functions to handle each event ############
	button0.on_clicked(roi_manager.decide_roi_class)
	button3.on_clicked(roi_manager.decrement_index)
	button4.on_clicked(roi_manager.increment_index)
	button6.on_clicked(roi_manager.terminate_gui)
	button7.on_clicked(roi_manager.erode_mask)
	button8.on_clicked(roi_manager.dilate_mask)
	fig.canvas.draw_idle()
	plt.show()

	############# The display ends with the upper right end end button on the graph. ##############
	final_roi = []

	for i in range(num_of_rois):
		roi = roi_info[i]
		if roi["CL"] >= 0:			
				final_roi.append(roi)

	###################### Export process of ROI information ########################
	img_path = pathlib.Path(img_path)
	img_stack, msk_stack = push_roi_areas(final_roi)	
	write_rois_in_stacks(img_path,dataset_dir,img_stack,msk_stack,final_roi)
	write_rois_in_json(img_path, dataset_dir, final_roi)

#END of main()
#######################################################################

if __name__ == '__main__':
	main( )


