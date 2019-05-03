# -*- coding: utf-8 -*-
# Module for loading .nxs files from SIXS beamline @ SOLEIL
# To be used together with the dictionnary 'alias_dict.txt'
# Code from Andrea Resta @ Soleil Synchrotron
# Modified 02052019 by Jerome Carnis @ CNRS IM2NP: removed unused functions
import tables
import numpy
import pickle

print('importing nxsReady')
print("'alias_dict.txt' path expected in 'specfile_name' parameter")
print("You can copy it from /Lib/site-packages/bcdi/preprocessing/")


class PrefParameters:
    # classe for preference parameters
    def __init__(self):
        self.namedisplays = list(("my suggestion", "short name", "short and family name", "full name", "nxs name"))
        self.inamedisplay = 0


class DataSet:
    """
    Dataset read the file and store it in an object, from this object we can retrieve the data to use it:
     MyFileObject=nxsRead.DataSet(path/filename, filename)
    """
    # define a classe to enter .nxs fiel-related parameters
    # the long name is the pathename 
    # short name should be the filenmane
    # alias_dict: the alias dictionnary, which should be located in the root directory of the experiment
    def __init__(self, longname, shortname, alias_dict, datafilter=None, pref=None, scan='FLY'):
        alias_dict = pickle.load(open(alias_dict, 'rb'))
        if pref is None:
            pref = PrefParameters()
        self.shortname = shortname
        self.THRESHOLD = 0
        if scan == 'FLY':
            shift = 1
        if scan == 'SBS':
            shift = 0
        if scan == 'HCS':
            shift = 1
        try:
            fichier = tables.open_file(longname, 'r')
            self.nodedatasizes = list()  # list of data array lengths
            for leaf in fichier.list_nodes('/')[0].scan_data:
                self.nodedatasizes.append(leaf.shape[0])
            self.npts = max(self.nodedatasizes)

            # we select only nodes of the same size, smaller arrays (e.g. of size 1) are let aside
            # here it generate the attributes of the DataSet class by defining their type
            self.nodenames = list()     # node names (ex data_01)
            self.nodelongnames = list()  # node comprehensible name AKA complete( ex: i14-c-cx2/ex/diff-uhv-k/position)
            self.nodenicknames = list()  # shortening of the long name AKA the last part of longname
            self.alias = []
            self.data = numpy.empty(0)   # empty table creation
            self.waveL = fichier.list_nodes('/')[0].SIXS.Monochromator.wavelength[0]
            self.energymono = fichier.list_nodes('/')[0].SIXS.Monochromator.energy[0]
            if fichier.list_nodes('/')[0].end_time.shape == ():
                self.end_time = fichier.list_nodes('/')[0].end_time.read().tostring()
            if fichier.list_nodes('/')[0].end_time.shape == (1,):
                self.end_time = fichier.list_nodes('/')[0].end_time[0]

            # here we assign the values to the attributes previously generated
            for leaf in fichier.list_nodes('/')[0].scan_data:
                nodelongname = ''
                nodenickname = ''
                if len(leaf.shape) == 1:
                    if leaf.shape[0] == self.npts:  
                        self.nodenames.append(leaf.name)

                        try:
                            nodelongname = leaf.attrs.long_name.decode('UTF-8')
                        except:
                            nodelongname = str(leaf).split()[0].split('/')[-1].split('_')[-1].lower()

                        if len(nodelongname) == 0:
                            nodelongname = leaf.name  # if no name keep nxs file name
                        self.nodelongnames.append(nodelongname)   
                        self.data = numpy.concatenate((self.data, leaf.read()[1:]))
                        # add data to numpy array and remove the first point
    
                        if pref.inamedisplay <= 1:
                            nodenickname = nodelongname.split('/')[-1]   # take just the last part of the longname
                            self.nodenicknames.append(nodenickname) 
                                
                        elif pref.inamedisplay == 2:
                            try:
                                namesplit = nodelongname.split("/")
                                nodenickname = namesplit[-2]+"/"+namesplit[-1]  # take the two last if possible
                                self.nodenicknames.append(nodenickname)
                            except:
                                self.nodenicknames.append(nodelongname)

                        elif pref.inamedisplay == 3:
                            self.nodenicknames.append(nodelongname)  # take the full long name
                
                        elif pref.inamedisplay == 4:
                            self.nodenicknames.append(leaf.name)  # take nxs file name

                    if alias_dict:
                        try:
                            alias = alias_dict[nodelongname.lower()]
                            if alias in self.alias:
                                alias += '#'
                            self.alias.append(alias)
                            self.__dict__[alias] = leaf.read()[shift:]
                        except:
                            self.alias.append(nodenickname)
                            self.__dict__[nodenickname] = leaf.read()[shift:]
                            pass

                elif len(leaf.shape) == 3:
                    if leaf.shape[1] == 1065:
                        if shift:
                            self.efilm = leaf[:-shift]
                        else:
                            self.efilm = leaf[:]  # Careful: process is different for HCS, SBS and FLY
                    if leaf.shape[1] == 240:
                        if shift:
                            self.xfilm = leaf[:-shift]
                        else:
                            self.xfilm = leaf[:]
                    if leaf.shape[1] == 516:
                        if shift:
                            self.mfilm = leaf[:-shift]
                        else:
                            self.mfilm = leaf[:]
                    pass
                        
        except ValueError:
            print("probleme le fichier ", longname, "est corrompu")
            self.npts = 0
            self.nmotors = 0
            self.mins = numpy.empty(0)
            self.maxs = numpy.empty(0)
            self.data = numpy.empty(0)
            fichier.close()
            return

        except tables.exceptions.NoSuchNodeError:
            print("probleme le fichier ", longname, "est corrompu")
            self.npts = 0
            self.nmotors = 0
            self.mins = numpy.empty(0)
            self.maxs = numpy.empty(0)
            self.data = numpy.empty(0)
            fichier.close()
            return

        else:
            fichier.close()

        self.npts = self.npts-1     # remove the 1st point that is uncorrect due to the operation strategy of simplescan
        self.nmotors = len(self.nodenames)   # number of columns kept

        # if display preferences are "my suggestion", we look for a name appearing several times
        # in this case we choose the longest name

        if pref.inamedisplay == 0:
            for i in range(self.nmotors-1):  # pas la peine de faire le dernier point!
            
                nickname = self.nodenicknames[i]

                if nickname in self.nodenicknames[i+1:]:   # item in double
                    nodelongname = self.nodelongnames[i]
                    namesplit = nodelongname.split("/")
                    try:
                        nodenickname = namesplit[-2]+"/"+namesplit[-1]   # take the two last
                    except:
                        nodenickname = nodelongname   # take the two last
                        
                    self.nodenicknames[i] = nodenickname
                   
                    j = i
                    try:
                        while 1:
                            j = self.nodenicknames.index(j+1)
                            self.nodenicknames[j]=nodenickname
                            # careful, it is not garanteed that nodenickname!=nickname
                    except ValueError:
                        pass
        
        self.data = self.data.reshape((self.nmotors, self.npts))
        test = numpy.any(self.data != 0, axis=0)  # if non-zero value, the condition is verified
        self.data = numpy.compress(test, self.data, axis=1)
      
        if datafilter is not None:
            # filter values while looking at the condition on the filter
            if datafilter.role is not "none" and datafilter.ifil > -1:
                if datafilter.irole == 1:
                    self.data = numpy.compress(self.data[datafilter.ifil] > datafilter.value, self.data, axis=1)
                elif datafilter.irole == 2:
                    self.data = numpy.compress(self.data[datafilter.ifil] < datafilter.value, self.data, axis=1)
                elif datafilter.irole == 3:
                    self.data = numpy.compress(self.data[datafilter.ifil] != datafilter.value, self.data, axis=1)
        
        self.npts = self.data.shape[1]   # number of points not totally null
        
        if self.npts == 0:   # no more points after filtering
            self.mins = numpy.zeros(self.nmotors)
            self.maxs = numpy.ones(self.nmotors)
        else:
            self.mins = numpy.amin(self.data, axis=1)   # boundary for each parameter
            self.maxs = numpy.amax(self.data, axis=1)

        self.namelist = self.alias
