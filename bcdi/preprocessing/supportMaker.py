# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Steven Leake, steven.leake@esrf.fr

import numpy as np
import h5py as h5
import sys

helptext = """
The concept is to be able to build a support from a set of defined planes these planes can be positioned based
on physical size (nm) if known or eventually made into a fancy tool with a 3D view of the support versus the data
so you can match fringes. It may also be interesting to consider how to make 3D supports from other characterisation
methods (SEM etc...).
"""

####################################################################
def AddPolyCen(array, center, planes):
  dims = array.shape

  griddims = []
  for d in dims:
    griddims.append( slice(0,d) )
  grid = np.ogrid[ griddims ] 

  for plane in planes:
    sum1 = np.zeros(dims)
    sum2 = 0
    for d in range(len(dims)):
      sum1 += plane[d]*(grid[d]-center[d])
      sum2 += plane[d]**2 
    array += (sum1 <= sum2) * 1

  return ((array >= len(planes))*1).astype(array.dtype)


####################################################################
def MakePoly(dims, planes):
  return make_poly(dims, planes)

####################################################################
def make_poly(dims, planes):
  cen = []
  array = np.zeros( dims )
  for dim in dims:
    cen.append(dim/2)
  return AddPolyCen(array, cen, planes)

####################################################################
def MakePolyCen(dims, center, planes):
  array = np.zeros( dims )
  return AddPolyCen(array, center, planes)


class supportMaker(object):
	"""
	
	A masking class for support creation
	
	:param : rawdata
	:param : x-ray wavelength 
	:param : detector_distance - sample to detector (m)
	:param : detector_pixel_size - 2D (m)
	:param : ang_step (degrees)
	:param : braggAng (degrees)
	:param : planes = array of planes [x,y,z]
	:param : planesDist = array of plane distance to origin (m)
	:param : voxel_size = set the voxel size to some arbitrary size, np.array([x,y,z]) (m) 
	
	"""
	def __init__(self, rawdata, wavelength = None, detector_distance = None, 
						detector_pixel_size = None, ang_step = None, braggAng = None, 
						planes = None, planesDist = None, voxel_size = None):
		# set all parameters

		self.rawdata = rawdata	
		self.wavelength = wavelength
		self.detDist = detector_distance
		self.detector_pixel_size = detector_pixel_size
		self.braggAng = braggAng
		self.ang_step = ang_step
		
		# create the support
		self.set_support(rawdata)
		if voxel_size is None:
			self.calc_voxel_size(wavelength, detector_distance, detector_pixel_size, ang_step, braggAng)
		else:
			self.set_voxel_size(voxel_size)
		
		print("voxel_size: ", self.vox_size)	
		self.set_planes(planes,planesDist)
		self.support = MakePoly(self.support.shape, self.scaled_planes)
		

	def set_support(self, rawdata):
		self.support = np.zeros_like(rawdata)
		
	def set_voxel_size(self, voxel_size):
		self.vox_size = voxel_size

	def set_planes(self,planes_list,plane_distance_origin_list):
		
		if len(planes_list)!=len(plane_distance_origin_list):
			print("the number of planes does not match the number of distances")
			sys.exit()
			
		self.planes = np.array(planes_list)
		self.planesDist = np.array(plane_distance_origin_list)
		
		# based on voxel size - scale distance to plane metres to pixels
		# convert existing plane distance 
		d_pix = np.sqrt(np.sum((self.planes)**2,axis=1))
		print('\nD_pix',d_pix)
		
		# convert existing plane distance to metres to user defined size with a scalefactor
		d_m = np.sqrt(np.sum((self.vox_size*self.planes)**2,axis=1))
		print('\nDM',d_m)
		
		sf = (self.planesDist.reshape(1,self.planes.shape[0])/d_m).reshape(self.planes.shape[0],1)
		self.scaled_planes = self.planes*sf
		
		print(self.planesDist.reshape(1,self.planes.shape[0]))
		print('\n###',self.scaled_planes)
		print('\n### scale factor:',sf)
		print('\n###',self.planes)
		print('\n###',self.rawdata.shape)
			
	def get_support(self,):
		return self.support
		
	def get_planes(self,):
		return self.planes
	
	def get_planesDist(self,):
		return self.planesDist

	def calc_voxel_size(self, wavelength = None, detDist = None, pixel_size = None, ang_step = None, braggAng = None):
		# use the experiment parameters to determine the voxel size		
		ss = self.support.shape
		
		# calculate the angular dimension first
		q = 4*np.pi*np.sin(np.deg2rad(braggAng))/wavelength
		deltaQ = q*np.deg2rad(ang_step)*ss[0]
		a = np.pi*2/deltaQ
		
		# add the detector dimensions - don't forget to reverse - but pixels tend to be symmetric
		pixel_size.reverse()
		self.vox_size = np.r_[a,wavelength*detDist/(np.array(ss[1:])*np.array(pixel_size))]
		
		print("Voxel dimensions: ", self.vox_size*1e9, " (nm)")
		
	
def generatePlanesCuboid(x,y,z):
	planes = 	np.array([
				[1,0,0],
				[-1,0,0],
				[0,1,0],
				[0,-1,0],
				[0,0,1],
				[0,0,-1],
				])
	planesDist = np.array([
				[x/2],
				[x/2],
				[y/2],
				[y/2],
				[z/2],
				[z/2],
				])	
	return 	planes, planesDist		
"""
data=MakePoly((64,64,64),((1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)))
outf=h5.File('test12.h5','w')
outf['poly']=data

data=MakePolyCen((64,64,64),(10,10,10),((1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)))
data=MakePolyCen((64,64,64),(10,10,10),((10.1,0,0), (0,10,0), (0,0,10), (-10,0,0), (0,-10,0), (0,0,-10)))
outf['polycen']=data
import scipy.fftpack as fft
data1=fft.fftn(np.complex64(data))
outf['poly_fft']=abs(data1)
outf['poly_fft_shift']=abs(fft.fftshift(data1))
outf.close()
"""

# cuboid
planes, planesDist= generatePlanesCuboid(800,800,100)

#tetrahedra

planes = np.array([
			[1,1,1],
			[-1,1,-1],
			[1,-1,-1],
			[-1,-1,1],
			])
planesDist = np.array([
			[300],
			[300],
			[300],
			[300],
			])
					

#equilateral prism
planes = 	np.array([
			[-1,np.sqrt(3)/2.,0],
			[1,np.sqrt(3)/2.,0],
			[0,-1,0],
			[0,0,1],
			[0,0,-1],
			])
planesDist = np.array([
			[150],
			[150],
			[150],
			[50],
			[50],
			])			
#planes, planesDist= generatePlanesCuboid(800,800,100)
planesDist = planesDist*1E-9

import transformations as tfs
alpha, beta, gamma = 0,18,0
origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]

Rx = tfs.rotation_matrix(np.deg2rad(alpha), xaxis)[:3,:3]
Ry = tfs.rotation_matrix(np.deg2rad(beta), yaxis)[:3,:3]
Rz = tfs.rotation_matrix(np.deg2rad(gamma), zaxis)[:3,:3]

def rot_planes(planes,rot):
	#tfs.concatenate_matrices(Rx,Ry,Rz)
	print(planes)
	rp = [np.dot(rot,v) for v in planes]

	npl = []
	[npl.append(p.tolist()) for p in rp]
	planes = np.array(npl)
	print(planes)
	return planes

planes = rot_planes(planes,Rz)	
planes = rot_planes(planes,Ry)
	
#rawdata = np.zeros((64,64,64))
# error on filename 2x3x10 binning
rawdata = np.load('/data/id01/inhouse/otherlightsources/2019_Diamond_I13/esrf-analysis/mask_bin3x3x10x1_sum_croppedraw.npz')['arr_0']
print(rawdata.shape)
wavelength = 12.39842/10.2*1E-10 
detector_distance = 2.9
detector_pixel_size = [10*55e-6,3*55e-6]
ang_step = 0.004*2
braggAng = 9
supportMaker = supportMaker(rawdata,
							wavelength, 
							detector_distance, 
							detector_pixel_size, 
							ang_step, 
							braggAng,
							planes,
							planesDist,
							voxel_size = np.array([10,10,10])*1E-9)	
								
support = supportMaker.get_support()

'''
# lazy rotation
import scipy.fftpack as fft
from scipy.ndimage.interpolation import rotate

support = supportMaker.get_support()
#rotate around z axis
support = rotate(support,25,(2,0),reshape=False)  # replace with Bragg angle*2
#rotate around y axis RH about y
#support = rotate(support,25,(1,0),reshape=False)  # replace with Bragg angle*2
#rotate around x axis
#support = rotate(support,25,(2,1),reshape=False)  # replace with Bragg angle*2


support[support>=0.1] = 1
support[support<0.1] = 0

'''

# save to npz
np.savez('support.npz',support)

# save 2hdf5
import scipy.fftpack as fft
with h5.File('support.h5','a') as outf:
	outf['poly'] = support
	data1 = fft.fftn(support)
	#outf['poly_fft'] = abs(data1)
	outf['poly_fft_shift'] = abs(fft.fftshift(data1))
	outf['rawdata'] = rawdata

#np.savez('supportFFT.npz',fft.ifftshift(fft.ifftn(support)))

# contact Mark and Guillaume 
