# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 10/2019 : class based on make_a_polygon.py from R. Harder
#                 
#		The concept is to be able to build a support from a set of defined planes
# 		these planes can be positioned based on physical size (nm) if known or eventually
#       made into a fancy tool with a 3D view of the support versus the data so you can match fringes
#       it may also be interesting to consider how to make 3D supports from other characterisation methods (SEM etc)
#
#
#       authors:
#         Steven Leake, steven.leake@esrf.fr

import numpy as np
import h5py as h5
import sys
import pdb

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
	:param : mask previously defined to improve, pixels to mask = 1
	:param : flatfield if available
	:param : hotpix array / list of points ((dim0,..dimn),(dim0,..dimn),...)if available
	
	"""
	def __init__(self, rawdata, wavelength = None, detector_distance = None, 
						detector_pixel_size = None, ang_step = None, braggAng = None, 
						planes = None, planesDist = None):
		# set all parameters

		self.rawdata = rawdata	
		self.wavelength = wavelength, 
		self.detDist = detector_distance, 
		self.detector_pixel_size = detector_pixel_size,
		self.braggAng = braggAng,
		self.ang_step = ang_step
		
		# create the support
		self.set_support(rawdata)
		self.set_voxel_size(wavelength, detector_distance, detector_pixel_size, ang_step, braggAng)
		self.set_planes(planes,planesDist)
		self.support = MakePoly(self.support.shape, self.scaled_planes)

	def set_support(self, rawdata):
		self.support = np.zeros_like(rawdata)

	def set_voxel_size(self, wavelength = None, detDist = None, pixel_size = None, ang_step = None, braggAng = None):
		ss = self.support.shape
		# calculate the angular dimension first
		q = 4*np.pi*np.sin(np.deg2rad(braggAng))/wavelength
		deltaQ = q*np.deg2rad(ang_step)*ss[0]
		a = np.pi*2/deltaQ
		# add the detector dimensions - don't forget to reverse - but pixels tend to be symmetric
		pixel_size.reverse()
		self.vox_size = np.r_[a,wavelength*detDist/(np.array(ss[1:])*np.array(pixel_size))]
		print("Voxel dimensions: ", self.vox_size*1e9, " (nm)")
		
	def set_planes(self,planes_list,plane_distance_origin_list):
		if len(planes_list)!=len(plane_distance_origin_list):
			print("the number of planes does not match the number of distances")
			sys.exit()
		self.planes = np.array(planes_list)
		self.planesDist = np.array(plane_distance_origin_list)
		
		# based on voxel size - scale distance to plane metres to pixels
		d_pix = np.sqrt(np.sum(self.planes**2,axis=1))
		#pdb.set_trace()
		d_m = np.sqrt(np.sum((self.vox_size*self.planes)**2,axis=1))
		sf = (self.planesDist.reshape(1,self.planes.shape[0])/d_m).reshape(self.planes.shape[0],1)
		self.scaled_planes = self.planes*sf
			
	def get_support(self,):
		return self.support
		
	def get_planes(self,):
		return self.planes
	
	def get_planesDist(self,):
		return self.planesDist
		
def generatePlanesCuboid(z,y,x):
	planes = 	np.array([
				[1,0,0],
				[-1,0,0],
				[0,1,0],
				[0,-1,0],
				[0,0,1],
				[0,0,-1],
				])
	planesDist = np.array([
				[z/2],
				[z/2],
				[y/2],
				[y/2],
				[x/2],
				[x/2],
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
planes = 	np.array([
			[1,0,0],
			[-1,0,0],
			[0,1,0],
			[0,-1,0],
			[0,0,1],
			[0,0,-1],
			])
planesDist = np.array([
			[50],
			[50],
			[100],
			[100],
			[200],
			[200],
			])

#tetrahedra
"""
planes = 	np.array([
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
			
"""						
planesDist = planesDist*1E-9
rawdata = np.zeros((64,64,64))
wavelength = 12.39842/8*1E-10 
detector_distance = 1
detector_pixel_size = [55e-6,55e-6]
ang_step = 0.01
braggAng = 10
supportMaker = supportMaker(rawdata,wavelength, detector_distance, detector_pixel_size, ang_step, braggAng,planes,planesDist)		

import scipy.fftpack as fft
with h5.File('support.h5','a') as outf:
	outf['poly'] = supportMaker.get_support()
	data1 = fft.fftn(np.complex64(supportMaker.get_support()))
	outf['poly_fft'] = abs(data1)
	outf['poly_fft_shift'] = abs(fft.fftshift(data1))

# contact Mark and Guillaume 
