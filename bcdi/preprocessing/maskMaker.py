# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 06/2019 : class based on prepare_mask_cdi.py
#                 
#       authors:
#         Steven Leake, steven.leake@esrf.fr

import hdf5plugin  # for P10, should be imported before h5py or PyTables
import xrayutilities as xu
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("Qt5Agg")  # "Qt5Agg" or "Qt4Agg" depending on the version of Qt installer, bug with Tk
import pathlib
import os
import scipy.signal  # for medfilt2d
import sys
from scipy.io import savemat
import tkinter as tk
from tkinter import filedialog
import collections
from functools import partial
from matplotlib.colors import LogNorm
import h5py as h5

class maskMaker(object):
	"""
	
	A masking class for IO of mask creation
	
	:param : rawdata to mask
	:param : mask previously defined to improve, pixels to mask = 1
	:param : flatfield if available
	:param : hotpix array / list of points ((dim0,..dimn),(dim0,..dimn),...)if available
	
	"""
	def __init__(self, data2mask, mask = None, aliens = None, flatfield = None, hotpix = None, detector = None):
		# load data and optionally apply flatfield to the data 
		# and add hot pixels to a mask
		self.set_data2mask(data2mask)
		self.set_mask(mask)
		self.set_flatfield(flatfield)		
		self.set_hotpix(hotpix)
		self.set_aliens(aliens)
		self.set_detector(detector)
	
	def set_data2mask(self,data2mask):
		self.data2mask = data2mask 
		self.d2ms = self.data2mask.shape
		self.d2mndims = len(self.d2ms)
		
	def set_flatfield(self,flatfield):
		if flatfield is not None:
			self.flatfield = flatfield
			if self.d2mndims == 3:
				for i in range(self.d2ms):
					self.data2mask[i,:,:] *= self.flatfield
			else:
				self.data2mask *= self.flatfield			
		
	def set_hotpix(self,hotpix):
		# hotpixel should be 2D array or list of points
		if isinstance(hotpix,collections.Sequence):
			self.hotpix = np.zeros(self.d2ms)
			# take sequence x,y,z indices
			if self.d2mndims == 3:
				for p in hotpix:
					self.hotpix[p[0],p[1],p[2]] = 1
			elif self.d2mndims == 2:
				for p in hotpix:
					self.hotpix[p[0],p[1]] = 1
				
		elif isinstance(hotpix,np.ndarray):
			if self.d2mndims == 3 and len(hotpix.shape)==2:
				tmp = np.zeros_like(self.data2mask)
				for i in range(self.d2ms[0]):
					tmp[i,:,:] = hotpix
				self.hotpix = np.copy(tmp)
			else: #dims equal
				self.hotpix = hotpix
		else:
			self.hotpix = np.zeros(self.d2ms)
		
	def set_aliens(self,aliens):
		if isinstance(aliens,collections.Sequence):
			self.aliens = np.zeros(self.d2ms)
			# take sequence x,y,z indices
			if self.d2mndims == 3:
				for p in aliens:
					self.aliens[p[0],p[1],p[2]] = 1
			elif self.d2mndims == 2:
				for p in aliens:
					self.aliens[p[0],p[1]] = 1
				
		elif isinstance(aliens,np.ndarray):
			self.aliens = aliens
		else:
			self.aliens = np.zeros(self.d2ms)

	def set_detector(self,detector):
		if isinstance(detector,collections.Sequence):
			self.detector = np.zeros(self.d2ms)
			# take sequence x,y,z indices
			if self.d2mndims == 3:
				for p in detector:
					self.detector[p[0],p[1],p[2]] = 1
			elif self.d2mndims == 2:
				for p in detector:
					self.detector[p[0],p[1]] = 1
				
		elif isinstance(detector,np.ndarray):
			self.hotpix = detector
		else:
			self.hotpix = np.zeros(self.d2ms)
	
	def set_mask(self,mask):
		if mask is not None:
			#apply mask
			self.ms = mask.shape
			if len(self.ms) == 2:
				if self.d2mndims == 3:
					tmpmask = np.zeros(self.d2ms)
					for i in range(self.d2ms):
						tmpmask[i,:,:] = self.mask
					self.mask = tmpmask
			elif len(self.ms) > self.d2mndims:
				print("mask has more dimensions than data")
			else:
				self.mask = mask 
		else:
			#create mask
			self.ms = self.d2ms
			self.mask = np.zeros(self.ms)	
		
	def get_data2mask(self,):
		return self.data2mask
		
	def get_flatfield(self,):
		return self.flatfield
		
	def get_hotpix(self,):
		if self.d2ms==3:
			tmp = self.hotpix.sum(axis=0)
			tmp[tmp>0] = 1
			return tmp
		else:
			return self.hotpix	

	def get_mask(self,):
		return self.mask
			
	def add_hotpix(self, hotpix):
		self.hotpix += hotpix

	
class maskInteraction(maskMaker):
	"""
	
	All tools to interact with a maskMaker object
	
	"""
	def __init__(self, data2mask, mask = None, aliens = None, flatfield = None, 
						hotpix = None, xy=[], flag_pause = False, 
						max_cbar = 0.5, flag_interact = True, 
						flag_aliens = True,flag_hotpix = True , mask_zero_event = True):
					
		super(maskInteraction,self).__init__(data2mask, 
											mask = mask, 
											flatfield = flatfield, 
											hotpix = hotpix)
		self.xy = xy
		self.xywidth = []
		self.xyflag = []
		self.flag_pause = flag_pause
		self.max_cbar = max_cbar
		self.startIdx = np.zeros(len(self.d2ms))
		self.currentIdx = np.zeros(len(self.d2ms))
		self.vmin = 0
		self.vmax = max_cbar 
		self.width=5
		self.idx=0
		self.fig_title = ''
		self.flag_hotpix = flag_hotpix
		self.flag_aliens = flag_aliens
		self.flag_add = True	
		self.flag_mask_zero_event = mask_zero_event	
		self.init_fig2mask(flag_interact=flag_interact, 
								flag_aliens=flag_aliens, 
								flag_hotpix=flag_hotpix)	

	def init_fig2mask(self, flag_interact = True, **kwargs):
		"""
		Initialise the figure to interact with the mask
		
		"""		
		# check the mask second
		# include abort with save
		
		self.tmpdata = np.copy(self.data2mask)
		
		if flag_interact:
			
			# remove hotpixels + (optional) deadPixels
			if self.flag_hotpix:
				print("remove_hotpixels")
				for self.dim in [0]: #,[self.tmpdata[self.idx,:,:],self.tmpdata[:,self.idx,:],self.tmpdata[:,:,self.idx]]):
					keypress = self.press_key2mask2D
					on_click = self.on_click
					self.width=0
					data = np.sum(self.tmpdata, axis=self.dim)
					self.mask = np.zeros_like(data) #
					self.mask = self.get_hotpix()[0,:,:]
					
					if self.flag_mask_zero_event:
						self.mask+=self.mask_zero_event()
						
					self.spawn_figure(data,keypress,on_click,background_plot='0.5')#,title='Remove hot pixels')
					self.set_hotpix(np.copy(self.mask)) 
					del self.fig_mask
			
			# remove aliens						
			if self.flag_aliens:
				print("remove_aliens")		
				self.mask = self.aliens+self.hotpix
				self.mask[self.mask>0] = 1
				for self.dim,data in zip([0,1,2],[self.tmpdata[self.idx,:,:],self.tmpdata[:,self.idx,:],self.tmpdata[:,:,self.idx]]):
					#for self.dim,data in zip([0],[self.tmpdata[self.idx,:,:]]):
					keypress = self.press_key2mask3D
					on_click = self.on_click
					self.spawn_figure(data,press_key=keypress,on_click=on_click,background_plot='0.5')#, title='remove aliens')
					del self.fig_mask

			

	def spawn_figure(self,data,press_key,on_click,background_plot='0.5',):
		self.idx=0
		self.fig_mask = plt.figure()
		#data[self.mask==1] = 0
		plt.imshow(data,norm=LogNorm())#, vmin = self.vmin, vmax = self.vmax)
		self.update_fig_title()
		plt.connect('key_press_event', press_key)
		plt.connect('button_press_event', on_click) # modifies mask and updates figure
		plt.title(self.fig_title)
		self.fig_mask.set_facecolor(background_plot)
		plt.show()
		
	def mask_zero_event(self,):
		tmp = self.data2mask.sum(axis=0)
		deadPixels = np.zeros_like(tmp)
		deadPixels[tmp==0] = 1
		print('Total dead pixels identified: ', deadPixels.sum())
		return deadPixels

	def update_fig2mask(self,):
		plt.clf()			

		# set data to plot
		#print(self.mask.shape)
		self.tmpdata = np.copy(self.data2mask)
		if self.d2mndims == 3 and len(self.mask.shape) == 3:
			if self.dim==0:
				self.plotdata = self.tmpdata[self.idx,:,:]
				self.tmpmask = self.mask[self.idx,:,:]
			elif self.dim==1:
				self.plotdata = self.tmpdata[:,self.idx,:]
				self.tmpmask = self.mask[:,self.idx,:]
			elif self.dim==2:
				self.plotdata = self.tmpdata[:,:,self.idx]
				self.tmpmask = self.mask[:,:,self.idx]

		# data is 3D and mask is 2D i.e hotpixels
		elif self.d2mndims == 3 and len(self.mask.shape)==2:
			self.plotdata = self.tmpdata.sum(axis=self.dim)
			self.tmpmask = self.mask
		elif self.d2mndims == 2:
			self.plotdata = self.tmpdata
			self.tmpmask = self.mask

		self.plotdata[self.tmpmask==1] = 0
				
		plt.title(self.fig_title)
		plt.imshow(self.plotdata,norm=LogNorm())#, self.vmin, self.vmax)
		plt.draw()
		
	def on_click(self,event):
		"""
		Function to interact with a plot, return the position of clicked pixel. If flag_pause==1 or
		if the mouse is out of plot axes, it will not register the click

		:param event: mouse click event
		:return: updated list of vertices which defines a polygon to be masked
		"""
		if not event.inaxes:
			return
		if not self.flag_pause:
			_x, _y = int(np.rint(event.xdata)), int(np.rint(event.ydata))
			self.xy.append([_x, _y])
			self.xywidth.append(self.width)
			self.xyflag.append(self.flag_add)
			self.modify_mask(flag_add = self.flag_add, pos=[_y, _x], width = self.width)
		return		
	
	def modify_mask(self, flag_add = True, pos = [0,0], width = 0):
		a2m_min = np.array(pos) - int(width/2)
		a2m_max = np.array(pos) + int(width/2) +1
		
		# sanity check limits
		if a2m_min[1] < 0:
			a2m_min[1] = 0
		if a2m_min[0] < 0:
			a2m_min[0] = 0
        
		"""
		if a2m_max[1] > maxvalaxis:
		    a2m_max[1] = 
        if a2m_max[0] > :
            a2m_max[0] =         
		"""    

        # set the value in mask
		if flag_add:
			maskVal = 1
		else:
			maskVal = 0
		
		#print(flag_add,)
		# apply to the mask
		if len(self.mask.shape) == 3:
			if self.dim==0:
				self.mask[self.idx,a2m_min[0]:a2m_max[0],a2m_min[1]:a2m_max[1]] = maskVal
			elif self.dim==1:
				self.mask[a2m_min[0]:a2m_max[0],self.idx,a2m_min[1]:a2m_max[1]] = maskVal
			elif self.dim==2:
				self.mask[a2m_min[0]:a2m_max[0],a2m_min[1]:a2m_max[1],self.idx] = maskVal
		if len(self.mask.shape) == 2:
			self.mask[a2m_min[0]:a2m_max[0],a2m_min[1]:a2m_max[1]] = maskVal
			 
		self.update_fig2mask()

	def load_previous_idx_mask(self,):
		idx = self.idx-1
		if self.dim==0:
			self.mask[self.idx,:,:] = self.mask[idx,:,:] 
		elif self.dim==1:
			self.mask[:,self.idx,:] = self.mask[:,idx,:]
		elif self.dim==2:
			self.mask[:,:,self.idx] = self.mask[:,:,idx]		
		self.update_fig2mask()
		
	def zero_current_idx(self,):
		if self.dim==0:
			self.mask[self.idx,:,:] = 0
		elif self.dim==1:
			self.mask[:,self.idx,:] = 0
		elif self.dim==2:
			self.mask[:,:,self.idx] = 0		
		self.update_fig2mask()		
		
		
	def update_fig_title(self, title = None):
		if title is None:
			title = "... press h for help ..."
		self.fig_title = title
		
	def press_key2mask2D(self, event):
		"""
		Interact with a plot for masking parasitic diffraction intensity or detector gaps

		:param event: button press event
		:return: updated data, mask and controls
		"""
		key = event.key
		title =  "x to pause/resume masking for pan/zoom \n"
		title += "m mask ; b unmask ; n undo; q quit ; \n"
		title += "up larger ; down smaller ; right darker ; left brighter"
		
		self.update_fig_title(title)


		if key == 'h':
			self.update_fig2mask()
								
		elif key == 'up':
			self.width += 1
			if self.width > 30:
				self.width = 30
			print('width: ', self.width)

		elif key == 'down':
			self.width-=1
			if self.width < 0:
				self.width = 0
			print('width: ', self.width)
			
		elif key == 'right':
			self.vmax += 1
			print('vmax: ', self.vmax)
			self.update_fig2mask()			

		elif key == 'left':
			self.vmax -= 1
			if self.vmax < 1:
				self.vmax = 1
			print('vmax: ', self.vmax)
			self.update_fig2mask()
						
		elif key == 'm':
			self.flag_add = True

		elif key == 'b':
			self.flag_add = False
			
		elif key == 'n':
			if not self.xyflag[-1]:
				tmp_flag_add = True
			elif self.xyflag[-1]:
				tmp_flag_add = False
				
			self.modify_mask(flag_add=tmp_flag_add, pos = self.xy[-1][::-1],width = self.xywidth[-1])
			self.xy = self.xy[:-1]
			self.xywidth = self.xywidth[:-1]
			self.xyflag = self.xyflag[:-1]

		elif key == 'x':
			if not self.flag_pause:
				self.flag_pause = True
				print('pause for pan/zoom')
			else:
				self.flag_pause = False
				print('resume masking')

		elif key == 'w':
			self.save2filedialog()

		elif key == 'q':
			plt.close(self.fig_mask)



	def press_key2mask3D(self, event):
		"""
		Interact with a plot for masking parasitic diffraction intensity or detector gaps

		:param event: button press event
		:return: updated data, mask and controls
		"""
		key = event.key
		title = " Frame: %i/%i \n"
		title += "m mask ; b unmask ; q quit ; u next frame ; z previous frame\n"
		title += "j load last mask ; k restart current mask " #right darker ; left brighter"
		self.update_fig_title(title)


		if key == 'h':
			self.update_fig2mask()
					
		elif key == 'u':
			self.idx += 1
			if self.idx >= self.d2ms[self.dim]:
				self.idx = 0
			self.update_fig_title(title%(self.idx+1,self.d2ms[self.dim]))
			self.update_fig2mask()
			
		elif key == 'z':
			self.idx -= 1
			if self.idx <= 0:
				self.idx = self.d2ms[self.dim]-1
			self.update_fig_title(title%(self.idx+1,self.d2ms[self.dim]))
			self.update_fig2mask()

		elif key == 'h':
			self.update_fig2mask()
			
		if key == 'up':
			self.width += 1
			if self.width > 30:
				self.width = 30
			print('width: ', self.width)
			
		elif key == 'down':
			self.width-=1
			if self.width < 0:
				self.width = 0
			print('width: ', self.width)

		elif key == 'j':
			self.load_previous_idx_mask()

		elif key == 'k':
			self.zero_current_idx()
			
		elif key == 'right':
			self.vmax += 1
			print('vmax: ', self.vmax)
			self.update_fig2mask()

		elif key == 'left':
			self.vmax -= 1
			if self.vmax < 1:
				self.vmax = 1
			print('vmax: ', self.vmax)
			self.update_fig2mask()
			
		elif key == 'm':
			self.flag_add = True

		elif key == 'b':
			self.flag_add = False
			
		elif key == 'n':
			if not self.xyflag[-1]:
				tmp_flag_add = True
			elif self.xyflag[-1]:
				tmp_flag_add = False
				
			self.modify_mask(flag_add=tmp_flag_add, pos = self.xy[-1][::-1],width = self.xywidth[-1])
			self.xy = self.xy[:-1]
			self.xywidth = self.xywidth[:-1]
			self.xyflag = self.xyflag[:-1]

		elif key == 'x':
			if not self.flag_pause:
				self.flag_pause = True
				print('pause for pan/zoom')
			else:
				self.flag_pause = False
				print('resume masking')

		elif key == 'w':
			self.save2filedialog()

		elif key == 'q':
			plt.close(self.fig_mask)
								
			
	def save2filedialog(self):
		#		root = tk.Tk()
		#		root.withdraw()
		#file_path = filedialog.askopenfilenames(initialdir=datadir,
		#												filetypes=[("NPZ", "*.npz"),
		#												("NPY", "*.npy"), ("CXI", "*.cxi"), ("HDF5", "*.h5")])

		master = tk.Tk()
		tk.Label(master, text="path").grid(row=0)
		tk.Label(master, text="filename").grid(row=1)

		self.e1 = tk.Entry(master)
		self.e2 = tk.Entry(master)
		self.e1.insert(10, os.getcwd())
		self.e2.insert(10, "somefilename")

		self.e1.grid(row=0, column=1)
		self.e2.grid(row=1, column=1)

		tk.Button(master, 
				  text='Quit', 
				  command=master.quit).grid(row=3, 
											column=0, 
											sticky=tk.W, 
											pady=4)
		tk.Button(master, text='Save', command=self.save_file).grid(row=3, 
																	   column=1, 
																	   sticky=tk.W, 
																	   pady=4)

		master.mainloop()

		tk.mainloop()
	
	def save_file(self,):
		outpath = os.path.join(self.e1.get(), self.e2.get())
		print('saving to outpath:',outpath)
		try:
			
			with h5.File(outpath,'a') as h5f:
				h5f['mask'] = self.mask
				h5f['hotpix'] = self.hotpix
				h5f['data'] = self.data2mask
				h5f.close()
				#print(self.mask.shape,self.hotpix.shape)
		except:
			print('FAILED ... please try again')
				
			
		self.e1.delete(0, tk.END)
		self.e2.delete(0, tk.END)
		
			
datapath = '/data/id01/inhouse/otherlightsources/2019_sixs/results/S263/pynxraw/S263_pynx_norm_128_252_224.npz'
data = np.load(datapath)['data']
mask = np.zeros_like(data)
hotpix = np.zeros_like(data)

maskMaker = maskMaker(data,mask=mask,hotpix=hotpix)
maskInteraction = maskInteraction(data,mask=mask,hotpix=hotpix)


"""
# save the data 
outpath = 'some_path.h5'
with h5.File(outpath,'a') as h5f:
	h5f['mask'] = maskInteraction.get_mask()
	h5f['data'] = maskInteraction.get_data2mask()
"""
	
# TODO:
# median filter pixels
# add nexus formatting for deadpixel values in mask array
# save to file dialog doesnt work	

