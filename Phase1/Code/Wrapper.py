#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Saket Seshadri Gudimetla Hanumath (UID - 116332293)
saketsgh@terpmail.umd.edu
M.Eng. Robotics
University of Maryland, College Park

Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import itertools
import glob
import os
from sklearn.cluster import KMeans

# defining constants 
PI = np.pi
EXP = np.exp
SQRT = np.sqrt
SQ = np.square 
COS = np.cos
SIN = np.sin 

##############################################################################
def load_images():
	os.chdir("../BSDS500/Images")
	images = []
	  
	for i in range(1, 11):
		image_name = str(i) + '.jpg'
		images.append(cv2.imread(image_name))
	
	os.chdir("../../Code")
	return images

def show_filt_imgs(img_list, filter_type):
	
	if(filter_type == 'L'):
		rows = 9
		cols = 6
		for i in range(48):
			plt.subplot(rows, cols, i+1)
			plt.axis('off')
			plt.imshow(img_list[:,:,i], cmap = 'gray')

	elif(filter_type == 'G'):
		rows = len(img_list)/5
		cols = 5
		for i, img in enumerate(img_list):
			plt.subplot(rows, cols, i+1)
			plt.axis('off')
			plt.imshow(img, cmap = 'gray')
	else:
		rows = 2
		cols = 16
		for i, img in enumerate(img_list):
			plt.subplot(rows, cols, i+1)
			plt.axis('off')
			plt.imshow(img, cmap = 'gray')

##############################################################################
# functions used for gaussian filter bank generation 

def derivative_of_gauss(sig, n, pts):

	var = sig**2
	g = []
	X = pts[0]
	Y = pts[1]

	for x, y in zip(X, Y):

		g.append(-y*EXP(-1*(x**2 + y**2)/(2*var)))

	g = np.asarray(g)
	g = g.reshape((n, n))
	g = g/(2*PI*var*var)

	return g

def Dog_filter_bank():
	
	scale = 2
	num_orient = 16
	# orientations = np.arange(0, 2*PI, 2*PI/num_orient)
	# sigma = [1, 3]
	sig = SQRT(2)**np.arange(1, scale+1)

	# kernel dimensions
	n = 7
	size = int(n-1)/2

	x = [np.arange(-size, size+1)]
	y = x
	[x,y] = np.meshgrid(x,y)
	pts = np.array([x.flatten(), y.flatten()])
	imgs = []

	for i in range(scale):
		for j in range(num_orient) :
			angle = (PI*2*j)/num_orient	
			rot_mat = np.array([[COS(angle),-SIN(angle)], 
								[SIN(angle), COS(angle)]])
			rot_pts = np.dot(rot_mat, pts)	
			filter = derivative_of_gauss(sig[i], n, rot_pts)
			imgs.append(filter)
			# plt.subplot(scale, num_orient, num_orient*(i)+j+1)
			# plt.axis('off')
			# plt.imshow(filter, cmap='gray')

	# plt.savefig("DoGFilterBank")
	# plt.show()

	return imgs

##############################################################################
# functions used for generation of LM Filter Bank

def LoG(pts, sig):
	x = pts[0]
	y = pts[1]
	var = sig**2
	e = SQ(x)+SQ(y)
	log = (1/SQRT(2*PI*var))*EXP(-e/(2*var))*(e - (2*var))/(var**2)

	return log

def gauss_deriv(x, order, sig):

	var = sig**2
	num = EXP(-1*SQ(x)/(2*var))
	gaussian = num/SQRT(2*PI*(var))
	
	if order == 1:
		gaussian = -1*gaussian*(x/var)
		return gaussian
	
	elif order == 2:
		gaussian = gaussian*((SQ(x)-(var))/(var**2))
		return gaussian

	else:
		return gaussian


def LM_filter_bank(LMS):

	# defining scales for LMS/LML
	if(LMS == 1) :
		scales_4 = [1, SQRT(2), 2, 2*SQRT(2)]
	else:
		scales_4 = [SQRT(2), 2, 2*SQRT(2), 4]
	scales_3 = scales_4[:3]

	# elongation
	elongation = 3

	# Parameters for 1st and 2nd order Gauss filters 
	num_orient = 6
	num_fst_order = (len(scales_3))*num_orient
	num_scnd_order = (len(scales_3))*num_orient
	# sig = SQRT(2)**np.arange(1, scale+1)

	# kernel dimensions
	filt_dim = 49
	size = int(filt_dim-1)/2

	# generating points
	x = [np.arange(-size, size+1)]
	y = x
	[x,y] = np.meshgrid(x,y)
	pts = np.array([x.flatten(), y.flatten()])

	# store the 1st and 2nd derivatives
	filter_list = np.zeros((filt_dim, filt_dim, 48))
	img_cnt = 0

	# looping through 36(18+18) 1st and 2nd derivative of gaussians
	for s in range(len(scales_3)):
		for o in range(num_orient):
			
			angle = (PI*o)/num_orient
			
			rot_mat = np.array([[COS(angle),-SIN(angle)], 
								[SIN(angle), COS(angle)]])
			rot_pts = np.dot(rot_mat, pts)	

			# first derivative of gaussian
			gx = gauss_deriv(rot_pts[0], 0, elongation*scales_3[s])
			gy = gauss_deriv(rot_pts[1], 1, scales_3[s])
			first_deriv = gx*gy
			first_deriv = np.reshape(first_deriv, (filt_dim, filt_dim))
			filter_list[:, :, img_cnt] = first_deriv

			# second derivative
			gx = gauss_deriv(rot_pts[0], 0, elongation*scales_3[s])
			gy = gauss_deriv(rot_pts[1], 2, scales_3[s])
			sec_deriv = gx*gy
			sec_deriv = np.reshape(sec_deriv, (filt_dim, filt_dim))
			filter_list[:, :, img_cnt + num_scnd_order] = sec_deriv			
			
			img_cnt+=1

	img_cnt = num_fst_order + num_scnd_order 
	
	# simple gaussian 
	for index, scale in enumerate(scales_4) :
		gx = gauss_deriv(pts[0], 0, scale)
		gy = gauss_deriv(pts[1], 0, scale)
		gaussian = gx*gy
		gaussian *= (SQRT(2*PI)*scale) 
		gaussian = np.reshape(gaussian, (filt_dim, filt_dim))
		filter_list[:, :, img_cnt] = gaussian	
		img_cnt+=1

	# laplacian of gaussian at basic scale
	for index, scale in enumerate(scales_4) :
		log = LoG(pts, scale)
		log = np.reshape(log, (filt_dim, filt_dim))
		filter_list[:, :, img_cnt] = log	
		img_cnt+=1

	# laplacian of gaussian at 3*scale
	for index, scale in enumerate(scales_4) :
		log = LoG(pts, 3*scale)
		log = np.reshape(log, (filt_dim, filt_dim))
		filter_list[:, :, img_cnt] = log	
		img_cnt+=1

	return filter_list

##############################################################################
# functions used for Gabor Filter Bank generation

def Gabor_filter_bank():

	gabor_ = list()
	filters = 15

	sigma, theta, lambda_, psi, gamma = [9, 13], 0.25, 7, 0., 1

	for k in sigma:
		scale_x = k
		scale_y = float(k) / gamma

		std_ = 3
		xmax = np.ceil(max(1, max(abs(std_ * scale_x * COS(theta)), abs(std_ * scale_y * SIN(theta)))))
		ymax = np.ceil(max(1, max(abs(std_ * scale_x * SIN(theta)), abs(std_ * scale_y * COS(theta)))))
		xmin = -xmax
		ymin = -ymax
		(y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

		x_theta = x * COS(theta) + y * SIN(theta)
		y_theta = -x * SIN(theta) + y * COS(theta)

		gab_ = EXP(-.5 * (x_theta ** 2 / scale_x ** 2 + y_theta ** 2 / scale_y ** 2)) * COS(2 * PI / lambda_ * x_theta + psi)
		angle = np.linspace(0, 360, filters)
		
		for i in range(filters):

			image_center = tuple(np.array(np.array(gab_).shape[1::-1]) / 2)
			rotation_matrix = cv2.getRotationMatrix2D(image_center, angle[i], 1.0)
			result = cv2.warpAffine(gab_, rotation_matrix, gab_.shape[1::-1], flags=cv2.INTER_LINEAR)

			gabor_.append(result)

	return gabor_

##############################################################################
# functions used for Half Discs Masks 

def half_disk(radius):
	hd = np.zeros((radius * 2, radius * 2))
	rad_sq = radius ** 2;
	for i in range(radius):
		m = (i - radius) ** 2
		for j in range(2 * radius):
			if m + (j - radius) ** 2 < rad_sq:
				hd[i, j] = 1
	return hd

def rotateImage(image, angle):
	image_center = tuple(np.array(np.array(image).shape[1::-1]) / 2)
	rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return result

#############################################################################
# functions to compute gradient 

def get_binary(img, bin_value):

	binary_img = img * 0

	for r in range(img.shape[0]):
		for c in range(img.shape[1]):

			if img[r, c] == bin_value:
				binary_img[r, c] = 1   
			else:
				binary_img[r, c] = 0

	return binary_img

def map_gradient(map, lmask, rmask, num_bins):

	grad = np.zeros((map.shape[0], map.shape[1], 24))
	map = map.astype(np.float64)

	for i in range(24):
		chi = np.zeros((map.shape))
		for j in range(1, num_bins):

			gmap = get_binary(map, j)
			g = cv2.filter2D(gmap, -1, lmask[i])
			h = cv2.filter2D(gmap, -1, rmask[i])
			chi = chi + ((g-h)**2) / (g+h+0.0001)
		grad[:, :, i] = chi

	return grad

#############################################################################

def get_canny_baseline(image_name):

	os.chdir("../BSDS500/CannyBaseline")
	cwd = os.getcwd()
	c = cv2.imread(image_name)
	plt.imshow(c)
	plt.title('Canny Baseline')
	plt.show()
	os.chdir("../../Code")

	return c

def get_sobel_baseline(image_name):

	os.chdir("../BSDS500/SobelBaseline")
	s = cv2.imread(image_name)
	plt.imshow(s)
	plt.title('Sobel Baseline')
	plt.show()
	os.chdir("../../Code")

	return s

##############################################################################

def main(): 

	images = load_images()
	# select img num to be displayed
	im_num = 6	
	image_name = str(im_num+1) + '.png'
	my_path = os.getcwd()	
	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	# Filters generation 
	dog_filters = Dog_filter_bank()
	show_filt_imgs(dog_filters, 'D')
	plt.savefig(my_path + "/results/Filters/" +'Dog.png')
	plt.show()
	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	lms_filter_list = LM_filter_bank(0)
	lml_filter_list = LM_filter_bank(1)
	show_filt_imgs(lms_filter_list, 'L')
	plt.savefig(my_path + "/results/Filters/" + 'LMS.png')
	plt.show()

	show_filt_imgs(lml_filter_list, 'L')
	plt.savefig(my_path + "/results/Filters/" + 'LML.png')
	plt.show()
	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	gb_filter_list = Gabor_filter_bank()
	show_filt_imgs(gb_filter_list, 'G')
	plt.savefig(my_path + "/results/Filters/" + 'Gabor.png')
	plt.show()
	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	orientation = np.arange(90, 360, 360 / 16)
	scales = np.asarray([5, 15, 20])
	lmask = []
	rmask = []
	sz = scales.size
	oz = 8

	for i in range(0, sz):
		halfd_ = half_disk(scales[i])
		for m in range(oz):
			mask_1 = rotateImage(halfd_, orientation[m])
			lmask.append(mask_1)
			mask_2 = rotateImage(mask_1, 180)
			rmask.append(mask_2)
			plt.subplot(sz * 2, oz, oz * 2 * (i) + m + 1)
			plt.axis('off')
			plt.imshow(mask_1, cmap='gray')
			plt.subplot(sz * 2, oz, oz * 2 * (i) + m + 1 + oz)
			plt.axis('off')
			plt.imshow(mask_2, cmap='gray')
	plt.savefig(my_path + "/results/Filters/" + 'HDMasks.png')
	plt.show()
	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""
	# combining all filters
	all_filters_bank = []
	
	for i, dog_imgs in enumerate(dog_filters):
		all_filters_bank.append(dog_imgs)

	_, _, lm_size = lms_filter_list.shape
	for i in range(lm_size):
		all_filters_bank.append(lms_filter_list[:, :, i])

	for i in range(lm_size):
		all_filters_bank.append(lml_filter_list[:, :, i])

	for i, gb_img in enumerate(gb_filter_list):
		all_filters_bank.append(gb_img)

	all_filters_bank = np.asarray(all_filters_bank)
	# print(np.shape(all_filters_bank))

	samp_img = cv2.cvtColor(images[im_num],cv2.COLOR_BGR2GRAY)
	data = np.zeros((samp_img.size, len(all_filters_bank)))

	for i in range(len(all_filters_bank)):
		conv_img = cv2.filter2D(samp_img, -1, all_filters_bank[i])
		conv_img = conv_img.reshape((1, samp_img.size))
		data[:,i] = conv_img
	"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""
	k_means = KMeans(n_clusters=64, n_init=4)
	k_means.fit(data)
	labels = k_means.labels_
	Tmap = np.reshape(labels,(samp_img.shape))
	plt.imshow(Tmap)
	plt.title('Tmap')
	plt.axis('off')
	plt.savefig(my_path + "/results/TextonMap/" + 'TextonMap_'+str(im_num+1)+'.png')
	plt.show()
	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	Tg = map_gradient(Tmap, lmask, rmask, 64)	
	Tg_mean = np.mean(Tg, axis=2)
	print(Tg_mean.shape)
	plt.imshow(Tg_mean)
	plt.title('Tg')
	plt.savefig(my_path + "/results/Tg/" + 'Tg_'+str(im_num+1)+'.png')
	plt.show()
	
	"""
	Generate Brightness Map
	Perform brightness binning 
	"""
	samp_img1 = cv2.cvtColor(images[im_num], cv2.COLOR_BGR2GRAY)
	samp_img2 = samp_img1.reshape(-1,1)
	
	k_means = KMeans(n_clusters=16, random_state=4)
	k_means.fit(samp_img2)
	labels = k_means.labels_

	Bmap = np.reshape(labels, samp_img1.shape)
	low = np.min(Bmap)
	high = np.max(Bmap)
	Bmap_f = 255*(Bmap-low)/np.float((high-low))
	plt.imshow(Bmap_f)
	plt.title('Bmap')
	plt.savefig(my_path + "/results/BrightnessMap/" + 'BrightnessMap_'+str(im_num+1)+'.png')
	plt.show()

	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	Bg = map_gradient(Bmap, lmask, rmask, 16)	
	Bg_mean = np.mean(Bg, axis=2)
	plt.imshow(Bg_mean)
	plt.title('Bg')
	plt.savefig(my_path + "/results/Bg/" + 'Bg_'+str(im_num+1)+'.png')
	plt.show()

	"""
	Generate Color Map
	Perform color binning or clustering
	"""
	samp_img3 = images[im_num] 
	im = samp_img3.reshape(-1, 3)
	k_means = KMeans(n_clusters=16, random_state=4)
	k_means.fit(im)
	labels = k_means.labels_
	
	Cmap = np.reshape(labels, (samp_img3.shape[0], samp_img3.shape[1]))
	plt.imshow(Cmap)
	plt.title('Cmap')
	plt.savefig(my_path + "/results/ColorMap/" + 'ColorMap_'+str(im_num+1)+'.png')
	plt.show()

	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	Cg = map_gradient(Cmap, lmask, rmask, 16)	
	Cg_mean = np.mean(Cg, axis=2)
	plt.imshow(Cg_mean)
	plt.title('Cg')
	plt.savefig(my_path + "/results/Cg/" + 'Cg_'+str(im_num+1)+'.png')
	plt.show()
	
	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""

	s = get_sobel_baseline(image_name)
	sgray = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
	
	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""
	c = get_canny_baseline(image_name)
	cgray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""
	w = 0.5
	pb = ((Tg_mean+Bg_mean+Cg_mean)/3)*(w*cgray+(1-w)*sgray)
	plt.imshow(pb, cmap='gray')
	plt.title('PbLite')
	plt.savefig(my_path + "/results/PbLite/" + 'PbLite_'+str(im_num+1)+'.png')
	plt.show()

if __name__ == '__main__':
    main()
 


