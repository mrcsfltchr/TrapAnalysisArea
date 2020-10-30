import numpy as np
from trapanalysis import TrapGetter
import tifffile as tf
from qtpy import QtWidgets, QtCore
from keras.models import load_model
import time
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu
from scipy.ndimage.morphology import distance_transform_edt
from skimage.feature import peak_local_max
from Watchbackvideos import livestream
import qimage2ndarray as qnd
from AnnotatedVidViewer import TrapViewer
import pandas as pd
from skimage.draw import circle
from BackgroundFinder import BackgroundFinder
from HeatPlotGenerator import HeatPlotGenerator
import sys
from scipy.ndimage import gaussian_filter1d as smooth
from cv2 import warpAffine, getRotationMatrix2D, boundingRect
import cv2
from numpy.core.defchararray import add

class Analyser(object):

    def __init__(self,path=None):

        #add deposit for a list of paths,trap coordinates for each video and their corresponding labels.
        self.videopaths =None
        self.videopos = ''
        self.traps_by_vid = {}
        self.labels_by_vid = {}
        self.multivid_frames = {}

        self.videopath = path
        self.frames = None


        #add class which handles background extraction

        self.bgfinder = BackgroundFinder()


        #data repositories

        self.intensitytrace = {}
        self.firstintensitytrace = {}

        self.secondintensitytrace = {}
        self.firstsecondintensitytrace = {}

        self.bg_sub_intensity_trace = {}
        self.bg_sub_firstintensity_trace = {}

        self.filtered_intensity_trace = {}
        self.filtered_first_intensity_trace = {}
        self.rffitrace = {}
        
        self.second_bg_si_trace = {}
        self.second_bg_fsi_trace = {}

        self.areatrace = {}
        self.firstareatrace = {}

        self.filtered_areatrace = {}
        self.filtered_firstareatrace = {}
        self.rffatrace = {}
        
        self.firstsecondareatrace = {}
        self.secondareatrace = {}

        self.areaerrors = {}
        self.centres = {}
        self.firstcentres = {}

        self.trapgetter = TrapGetter()


        self.persisting_times = {}

        self.visibletrapframe = None

        self.classifier = load_model('VesClassifier')
        self.mask = None
        self.t0frameNo = 0

        self.vesiclelife = None

        self.missing_peaks = {}

        self.livestream = None
        self.duplicates = np.array([])

        self.bgintens = None
        self.heat_data = np.array([])
        self.HPG = HeatPlotGenerator()


    def load_frames(self,t0 = None,tmax = None):

        '''
        with tf.TiffFile(self.videopath) as tif:
            if t0 is not None:
                frames = tif.asarray(key = slice(t0,tmax))
            else:
                frames = tif.asarray()
        print('Done!')
        '''
        
        frames = tf.TiffFile(self.videopath)
        
        try:
            videolength = frames.imagej_metadata['frames']
        except:
            videolength = None
            
        return frames, videolength





    def get_traps(self,drug_start_frame, alternateframe = None,alternateframe_index = None, threshold = None):
        
        print(drug_start_frame)

        #drug_start_frame is the index of the frame in which drug arrives. If an alternateframe is supplied, the index of it must also be supplied. This is compared to the drug_start_frame index. If it is larger we then don't determine a new threshold to binarise the alternate frame as we assume drug has arrived. When drug has arrived otsu's thresholding will fail.
        if self.frames is not None and alternateframe is None:
            self.vframe = self.frames.asarray(key = int(drug_start_frame))
            print(self.vframe)
            
            
        '''
        self.visibletrapframe = self.frames[-1]

        self.trapgetter.get_trap_positions(self.visibletrapframe)
        '''

        
        if alternateframe is not None and alternateframe_index is not None:
            if alternateframe_index > drug_start_frame:
                self.trapgetter.get_vesicle_positions(alternateframe,True)
            else:
                self.trapgetter.get_vesicle_positions(alternateframe)
           
        elif alternateframe is not None and threshold is not None:
            self.trapgetter.get_vesicle_positions(alternateframe,threshold = threshold)
        else:   
            
            
            
            self.trapgetter.get_vesicle_positions(self.vframe)
            
        traps,labels = self.trapgetter.remove_duplicates()

        '''
        if self.videopath.find('Pos')+3 == '0':
            self.analyser.videopos = '9'
        else:    
            self.videopos = self.videopath[self.videopath.find('Pos')+3]
        print(self.videopos)
        prelabels = int(self.videopos) * np.ones(labels.shape).astype(int)
        prelabels = prelabels.astype(str)
        labels = labels.astype(str)
        print(labels.shape[0])
        labels = add(prelabels,labels)
        
        labels = labels.astype(int)
        '''
        self.trapgetter.labels = labels
        
        return traps,labels

    def rectangle(self,start, end=None, extent=None, shape=None):

        if extent is not None:
            end = np.array(start) + np.array(extent)
        elif end is None:
            raise ValueError("Either `end` or `extent` must be given")
        tl = np.minimum(start, end)
        br = np.maximum(start, end)
        if extent is None:
            br += 1
        if shape is not None:
           br = np.minimum(shape, br)
           tl = np.maximum(np.zeros_like(shape), tl)
        coords = np.meshgrid(*[np.arange(st, en) for st, en in zip(tuple(tl),
                                                           tuple(br))])

        return np.vstack((coords[0].flatten(),coords[1].flatten()))

    def get_clips(self):

        counter = 0
        for trap in self.trapgetter.trap_positions:

            if self.mask is not None:

                try:
                    self.mask = np.vstack((self.mask,self.rectangle(start = trap-[self.trapgetter.topboxrel,self.trapgetter.leftboxrel],end = trap +[self.trapgetter.bottomboxrel,self.trapgetter.rightboxrel])))
                except:
                    continue
            else:
                    self.mask = self.rectangle(start = trap-[self.trapgetter.topboxrel,self.trapgetter.leftboxrel],end = trap +[self.trapgetter.bottomboxrel,self.trapgetter.rightboxrel])

            counter +=1
        print(counter)
        self.mask = self.mask.reshape(int(self.mask.shape[0]/2),2,self.mask.shape[1])

    def get_clips_alt(self):

        print("list of trap coords has shape, ",self.trapgetter.trap_positions.shape[0])
        #initialize label variable as the label of the first trap.

        self.mask = None
        for trap in self.trapgetter.trap_positions:


            clip = np.zeros_like(self.frames.asarray(key = 0))

            try:
                clip[self.rectangle(start = trap-[self.trapgetter.topboxrel,self.trapgetter.leftboxrel],end = trap +[self.trapgetter.bottomboxrel,self.trapgetter.rightboxrel])[0],self.rectangle(start = trap-[self.trapgetter.topboxrel,self.trapgetter.leftboxrel],end = trap +[self.trapgetter.bottomboxrel,self.trapgetter.rightboxrel])[1]]=1
                print('after fitting rectangle ',clip.shape[0])
            except:
                #if clip cannot be made, for example if the trap centre is too close to the edge of the frame that a rectangle box of the defined size cannot be extracted from the image. Rather than handle this by changing the box size automatically, we just bin the boxes that are too close to the edge.

                #it is necessary to delete the labels of binned boxes and the trap position from future trap positions.

                print('something went wrong here')
                distances = np.linalg.norm(self.trapgetter.trap_positions -trap,axis = 1)


                self.trapgetter.labels = self.trapgetter.labels[distances > 1e-8]
                self.trapgetter.trap_positions = self.trapgetter.trap_positions[distances > 1e-8,:]

                continue
            if self.mask is not None:
                self.mask = np.vstack((self.mask,clip.flatten()))

            else:
                self.mask = clip.flatten()


        
        print("mask has shape, ",self.mask.shape[0])
        if self.mask.shape[0] == 512*512:
            self.mask = self.mask.reshape(1,512*512)
    def sett0frame(self,frameno):

        self.t0frameNo = int(frameno)
    def classify_clips(self,multivid = False):

        #classify contents of boxes in t = 0 frame, and then switch off recording for initially empty boxes
        if multivid:
            initial_frame = self.frames.asarray(key = 0)
        else:
            initial_frame = self.frames.asarray(key = self.t0frameNo)
            


        self.clips = initial_frame.flatten().T*self.mask

        self.clips = self.clips[self.clips >0].reshape(self.clips.shape[0],31,31)

        self.clips -= np.min(self.clips)

        
        self.clips = self.clips/np.max(self.clips)

        self.class_labels = self.classifier.predict(self.clips[:,:30,:30,np.newaxis],batch_size = self.clips.shape[0])

        
        self.class_labels = self.class_labels.reshape(self.class_labels.shape[0],)
        print("class labels length, ",len(self.class_labels))

        self.active_labels = self.trapgetter.labels[self.class_labels.astype(int) == 1]
        self.activemask = self.mask[self.class_labels.astype(int) == 1]

    def process_clips(self,clips):
        #deprecated. Included in function classify_clips

        clips -= np.min(clips)

        clips = clips/np.max(clips)

        return clips


    def analyse_frames(self,maxframe,multivid = False,just_area=False):



        #initialize active labels and active mask here.
        kernel = np.ones((3,3),np.uint8)
        if multivid:
            self.t0frameNo = 0
            maxframe = self.videolength
        else:
            self.videolength = maxframe
        print(maxframe)
        counter = 0
        for ind in range(self.t0frameNo,maxframe):
            frame = self.frames.asarray(key = ind)
            
            if not counter % 10 :
                print(str(counter) + ' frames analysed')
                
            tic = time.time()



            self.clips = frame.flatten().T*self.activemask
            if counter == 0:
                # once the classifier has identified what it believes are the only boxes with vesicles inside at t0 the fates of these boxes are sealed. The contents of these boxes will be passed to the thresholding function and a potential vesicle centre identified. This is done in parallel and independently of the intensities recorded within the process which reclassifies the box contents in every frame.




                self.firstactivemask = self.activemask
                self.firstactivelabels = self.active_labels

            if self.clips.shape[0] == 0:
                
                
                self.firstactiveclips = frame.flatten().T*self.firstactivemask
                self.firstactiveclips = self.firstactiveclips[self.firstactiveclips > 0].reshape(self.firstactiveclips.shape[0],31,31)
            
                if just_area:
                    self.run_just_area_analysis()
                else:
                    self.extract_intens_all_ves(counter,kernel)
                counter +=1
                
                continue
            
            
            self.clips = self.clips[self.clips >0].reshape(self.clips.shape[0],31,31)




            self.firstactiveclips = frame.flatten().T*self.firstactivemask
            self.firstactiveclips = self.firstactiveclips[self.firstactiveclips > 0].reshape(self.firstactiveclips.shape[0],31,31)

            self.zerominclips = self.clips - np.min(self.clips)

            self.zero2oneclips = self.zerominclips/np.max(self.zerominclips)

            self.class_labels = self.classifier.predict(self.zero2oneclips[:,:30,:30,np.newaxis],batch_size = self.clips.shape[0])

            self.class_labels = self.class_labels.reshape(self.class_labels.shape[0],)
            if just_area:
                self.run_just_area_analysis()
            else:
                
                self.extract_intens_all_ves(counter,kernel)


            self.active_labels = self.active_labels[self.class_labels.astype(int) == 1]



            self.activemask = self.activemask[self.class_labels.astype(int) == 1]

            toc = time.time()


            counter +=1
    def run_just_area_analysis(self):
        
        kernel = np.ones((3,3),np.uint8)
        plt.figure()
        for label in self.firstactivelabels:
            if self.clips.shape[0] > 0:
            
                clip = self.clips[self.active_labels == label]
                if clip.shape[0] > 0:
                    threshold = threshold_otsu(clip)
            
                    testclip = np.zeros_like(self.clips[self.active_labels == label].reshape(31,31))
            
                    testclip[self.clips[self.active_labels == label].reshape(31,31) > (threshold + 0.3*threshold)] = 1
                    
                    testclip = cv2.normalize(src=testclip,dst=None,alpha = 0,beta = 255,norm_type = cv2.NORM_MINMAX,dtype = cv2.CV_8UC1)
                    
                    opening = cv2.morphologyEx(testclip,cv2.MORPH_OPEN,kernel,iterations = 2)
                    
                    #sure background area
                    sure_bg = cv2.dilate(opening,kernel,iterations=3)
                    
                    #Find sure foreground area
                    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
                    
                    ret, sure_fg = cv2.threshold(dist_transform,0.6*dist_transform.max(),255,0)
                    sure_fg = np.uint8(sure_fg)
                    unknown = cv2.subtract(sure_bg,sure_fg)
                    
                    ret, markers = cv2.connectedComponents(sure_fg)
                    markers = markers +1
                    markers[unknown==255] = 0
                    clip = clip.reshape(31,31)
                    clip = smooth(clip,sigma = 1)
                    
                    print(clip.shape, clip)
                    nclip = cv2.normalize(src=clip,dst=None,alpha = 0,beta = 255,norm_type = cv2.NORM_MINMAX,dtype = cv2.CV_8UC1)
                    gclip = cv2.cvtColor(nclip,cv2.COLOR_GRAY2BGR)
                    markers = cv2.watershed(gclip,markers)
                    print(np.argwhere(markers == 2))
                    
                    
                    
                    '''gclip[markers == -1] = 255
                    gclip[markers==3] = 125
                    gclip[markers==1]= 0
                    plt.subplot(121)
                    plt.imshow(gclip,cmap = 'gray')
                    plt.subplot(122)
                    plt.imshow(clip)
                    plt.show()
                    '''
    def count_foreground_pixels(self,foreground):
        #print(foreground[foreground > 0].shape[0])
        return foreground[foreground > 0].shape[0]
    
                     
    def extract_area(self,clip):
        # this function takes in a clip of the image, unbinarised
        kernel = np.ones((3,3),np.uint8)
        #kernel is the 2D spreading function which dilates the edges in a process similar to convolution but where the convolution is performed conditionally on a foreground pixel not being completely surrounded by other foreground pixels
        clip = clip.reshape(31,31)
        #testclip = cv2.normalize(src=testclip,dst=None,alpha = 0,beta = 255,norm_type = cv2.NORM_MINMAX,dtype = cv2.CV_8UC1)
        clip8 = (clip/256).astype('uint8')     
        ret, thresh = cv2.threshold(clip8,0,255,cv2.THRESH_OTSU)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations = 2)
        
        #sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)
        
        #Find sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        
        ret, sure_fg = cv2.threshold(dist_transform,0.6*dist_transform.max(),255,0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        
        ret, markers = cv2.connectedComponents(sure_fg,ltype = cv2.CV_32S)
        markers = markers +1
        markers[unknown==255] = 0
        
        clip8 = smooth(clip8,sigma =1)
        
        #plt.imshow(clip8)
        #plt.show()
        #nclip = cv2.normalize(src=clip,dst=None,alpha = 0,beta = 255,norm_type = cv2.NORM_MINMAX,dtype = cv2.CV_8UC1)
        gclip = cv2.cvtColor(clip8,cv2.COLOR_GRAY2RGB)
        markers = cv2.watershed(gclip,markers)
        
        #Update code here to decide sensibly which 
        pixelarea = markers[markers ==2].shape[0]
        
        
        return pixelarea
    
    def extract_intens_all_ves(self,counter,kernel):
        
        for label in self.firstactivelabels:
            self.extract_intensity(label,counter,kernel)
            
    def visualise_active_clips(self,num = 3):

        number_of_samples = num

        idxs = np.arange(self.active_labels.shape[0])
        if idxs.shape[0] >0:
            idxs = np.random.choice(idxs, number_of_samples, replace=False)
            for i, idx in enumerate(idxs):
                plt.subplot(number_of_samples, 1, i+1)
                plt.imshow(self.clips[idx],cmap = 'gray')

                plt.xlabel('Active Clip')


                if i == 0:
                    plt.title('Active Boxes detected by Classifier')

        plt.tight_layout()
        plt.show()

    ########## DEPRECATED ##########
    #def extract_background_intensity(self):
    #    self.bgoffset = [int(0.5*np.median(self.trapgetter.distances[np.arange(self.trapgetter.distances.shape[0]),self.trapgetter.sorted_distances[:,1]][self.trapgetter.distances[np.arange(self.trapgetter.distances.shape[0]),self.trapgetter.sorted_distances[:,1]] > 10])),0]
    #    self.bgcentrecoords = self.trapgetter.trap_positions[self.trapgetter.labels == 0][0] - self.bgoffset
    #    last_few_frames = self.frames.asarray(slice(self.videolength-5,self.videolength))
    #    self.bgintens = np.average(last_few_frames[:,self.bgcentrecoords[0]-3:self.bgcentrecoords[0]+3,self.bgcentrecoords[1]-3:self.bgcentrecoords[1]+3],axis = (1,2))

    def extract_background(self,maxlen,borstel=False,pH = False):
        
        last_few_frames = self.frames.asarray(slice(maxlen-5,maxlen))
        if borstel:
            self.bgintens = 3750
        elif pH:
            # write last_few_frames = 
        else:
            self.bgintens = np.average(last_few_frames)



    def extract_intensity(self,label,counter,kernel):

        
        if self.clips.shape[0] > 0:
            
            clip = self.clips[self.active_labels == label]
            if clip.shape[0] > 0:
                threshold = threshold_otsu(clip)
        
                testclip = np.zeros_like(self.clips[self.active_labels == label].reshape(31,31))
        
                testclip[self.clips[self.active_labels == label].reshape(31,31) > threshold] = 1
        
                try:
                    self.secondareatrace[str(label)].append(len(testclip[testclip >0]))
                except KeyError:
                    self.secondareatrace[str(label)] = [len(testclip[testclip >0])]
        
                dt = distance_transform_edt(testclip)
        
                try:
                    centre = list(peak_local_max(dt,threshold_rel = 0.6)[0])
        
                    rr,cc = circle(centre[0],centre[1],int(dt[centre[0],centre[1]]),shape = dt.shape)
                    img = np.zeros_like(dt)
                    img[rr,cc] = self.clips[self.active_labels == label][0][rr,cc]
        
                    try:
                        #pixelcount = self.extract_area(clip)
                
                        #self.firstareatrace[str(label)].append(pixelcount)
                        #self.areatrace[str(label)].append(self.extract_area(testclip,kernel,clip))
                        #self.filtered_areatrace[str(label)] = smooth(self.firstareatrace[str(label)],3)
        
                        self.secondintensitytrace[str(label)].append(np.average(img[img > 0]))
                    except KeyError:
                        
                        #pixelcount = self.extract_area(clip)
                
                        #self.firstareatrace[str(label)]=[pixelcount]                       
                        #self.areatrace[str(label)] = [self.extract_area(testclip,kernel,clip)]
                        #self.filtered_areatrace[str(label)]=smooth(self.firstareatrace[str(label)],3)
                        self.secondintensitytrace[str(label)] = [np.average(img[img>0])]
        
        
                except IndexError:
                    centre = []
                    try:
                        self.missing_peaks[str(label)].append(counter)
                    except KeyError:
                        self.missing_peaks[str(label)] = [counter]
        
        
        
                if len(centre) >0:
        
        
                    x1 = 4
                    y1 = 4
                    lx = 31
                    ly = 31
                    #check the box can fit into clip
        
                    centre = np.array(centre)
                    dims_min= np.array([x1,y1])
                    dims_max = np.array([lx-x1,ly-y1])
        
                    margins_from_edge = np.vstack(((centre-dims_min),(dims_max - centre)))
                    if np.any(margins_from_edge.flatten() < 0):
                        x1 += np.min(margins_from_edge)
                        y1 += np.min(margins_from_edge)
        
        
                    small_box_in_ves = self.clips[self.active_labels == label][0][centre[0]-y1:centre[0]+y1,centre[1]-x1:centre[1]+x1]
        
                    av_intens = np.average(small_box_in_ves)
        
                    try:
        
                        self.intensitytrace[str(label)].append(av_intens)
        
        
                    except KeyError:
        
                        self.intensitytrace[str(label)] = [av_intens]
        
                    try:
                        self.centres[str(label)].append(centre)
                    except KeyError:
                        self.centres[str(label)] = [centre]
        
        
        firstclip = self.firstactiveclips[self.firstactivelabels == label]
        threshold = threshold_otsu(firstclip)
        testclip = np.zeros_like(firstclip.reshape(31,31))
        testclip[firstclip.reshape(31,31) > threshold] = 1


        try:
            self.firstsecondareatrace[str(label)].append(self.extract_area(firstclip))
        except KeyError:
            self.firstsecondareatrace[str(label)] = [self.extract_area(firstclip)]

        dt = distance_transform_edt(testclip)

        try:
            #if thresholding has failed completely to find a foreground, no peak can be found. The return value assigned to
            #the variable 'centre' is not an array so cannot be indexed. An index error will be thrown and caught below

            centre = list(peak_local_max(dt,threshold_rel = 0.6)[0])

            rr,cc = circle(centre[0],centre[1],int(dt[centre[0],centre[1]]),shape = dt.shape)
            img = np.zeros_like(dt)
            img[rr,cc] = firstclip[0][rr,cc]

            try:
                pixelcount = self.extract_area(firstclip)
                
                self.firstareatrace[str(label)].append(pixelcount)            
                self.areatrace[str(label)].append(self.extract_area(firstclip))
                self.filtered_firstareatrace[str(label)] = smooth(self.firstareatrace[str(label)],3)

                self.firstsecondintensitytrace[str(label)].append(np.average(img[img > 0]))
            except KeyError:
                
                pixelcount = self.extract_area(firstclip)
                self.areatrace[str(label)] = [self.extract_area(firstclip)]
                self.firstareatrace[str(label)]= [pixelcount]
                
                self.filtered_firstareatrace[str(label)]=smooth(self.firstareatrace[str(label)],3)
                self.firstsecondintensitytrace[str(label)] = [np.average(img[img>0])]


        except IndexError:

            #If no centre position is found, we take the last centre position found and take the intensity value from this
            #If thresholding fails to find a foreground in the first frame, such that there is no previously recorded centre position
            #we do not record any intensity or area values for the contents of this box in this frame.


            try:

                centre = self.firstcentres[str(label)][-1]
                self.if_threshold_fails(centre,label,firstclip,testclip)

            except KeyError or IndexError:

                centre = []


            '''
            centre = []
            try:
                self.missing_peaks[str(label)].append(counter)
            except KeyError:
                self.missing_peaks[str(label)] = [counter]
            '''




        if len(centre) >0:



            x1 = 4
            y1 = 4
            lx = 31
            ly = 31
            #check the box can fit into clip

            centre = np.array(centre)
            dims_min= np.array([x1,y1])
            dims_max = np.array([lx-x1,ly-y1])

            margins_from_edge = np.vstack(((centre-dims_min),(dims_max - centre)))
            if np.any(margins_from_edge.flatten() < 0):
                x1 += np.min(margins_from_edge)
                y1 += np.min(margins_from_edge)


            small_box_in_ves = firstclip[0][centre[0]-y1:centre[0]+y1,centre[1]-x1:centre[1]+x1]

            av_intens = np.average(small_box_in_ves)

            try:

                self.firstintensitytrace[str(label)].append(av_intens)


            except KeyError:

                self.firstintensitytrace[str(label)] = [av_intens]

            try:
                self.firstcentres[str(label)].append(centre)
            except KeyError:
                self.firstcentres[str(label)] = [centre]


    def get_eccentricity(self, centre,clip):
        #This rotates the clip around angles up to 90 degrees rotation and fits a bounding box, to find the maximum and minimum diameters of deformed vesicle
        aspect_ratios = []
        
        for i in range(0,90,5):
        
            rotate_matrix = getRotationMatrix2D(center = (centre[1],centre[0]),angle = i,scale = 1)
            
            rotated_clip = warpAffine(src = clip,M = rotate_matrix, dsize = (31,31))
        
            coords = np.nonzero(rotated_clip)
            coords = np.vstack(coords)
            coords = coords.T
            
            
            
            x,y,wx,wy  = boundingRect(coords)

            
            
            
            
            if wy > wx and wy> 0 and wx > 0:
                aspect_ratios.append(wy/wx)
            elif wy > 0 and wx > 0:
                aspect_ratios.append(wx/wy)
                
            else: 
                aspect_ratios.append(0)
                
        
        return max(aspect_ratios)
    
    
    

    def if_threshold_fails(self,centre,label,firstclip,testclip):


        rr,cc = circle(centre[0],centre[1],int(np.sqrt(self.firstareatrace[str(label)][-1])),shape = firstclip.shape)
        img = np.zeros_like(firstclip[0])
        img[rr,cc] = firstclip[0][rr,cc]
        #eccentricity = self.get_eccentricity(centre,testclip)
        try:

            
            self.firstareatrace[str(label)].append(0)
            self.filtered_firstareatrace[str(label)] = smooth(self.firstareatrace[str(label)],3)

            self.firstsecondintensitytrace[str(label)].append(np.average(img[img > 0]))
        except KeyError:
            self.firstareatrace[str(label)] = [0]
            self.filtered_firstareatrace[str(label)]=smooth(self.firstareatrace[str(label)],3)
            self.firstsecondintensitytrace[str(label)] = [np.average(img[img>0])]

    def delete_vesicle(self,label_as_str):


        if self.bg_sub_intensity_trace[label_as_str] is not None:

            del self.bg_sub_intensity_trace[label_as_str]
            del self.filtered_intensity_trace[label_as_str]
            del self.bg_sub_firstintensity_trace[label_as_str]
            del self.filtered_first_intensity_trace[label_as_str]
            del self.second_bg_si_trace[label_as_str]
            del self.second_bg_fsi_trace[label_as_str]


        elif self.intensitytrace[label_as_str] is not None:

            del self.intensitytrace[label_as_str]
            del self.firstintensitytrace[label_as_str]

        del self.areatrace[label_as_str]
        del self.firstareatrace[label_as_str]
        del self.filtered_areatrace[label_as_str]
        del self.filtered_firstareatrace[label_as_str]

        del self.centres[label_as_str]
        del self.firstcentres[label_as_str]






    def visualise_box_contents(self,label):

        trap = self.trapgetter.trap_positions[self.trapgetter.labels == label][0]


        clip = np.zeros_like(self.frames[0])

        try:
            clip[self.rectangle(start = trap-[self.trapgetter.topboxrel,self.trapgetter.leftboxrel],end = trap +[self.trapgetter.bottomboxrel,self.trapgetter.rightboxrel])[0],self.rectangle(start = trap-[self.trapgetter.topboxrel,self.trapgetter.leftboxrel],end = trap +[self.trapgetter.bottomboxrel,self.trapgetter.rightboxrel])[1]]=1

        except:
            raise IndexError


        self.vesiclelife = self.frames.reshape(self.frames.shape[0],self.frames.shape[1]*self.frames.shape[2])*clip.flatten()

        self.vesiclelife = self.vesiclelife[self.vesiclelife !=0]

        self.vesiclelife = self.vesiclelife.reshape(self.frames.shape[0],31,31)


    def plotnow(self,label):
        self.plotIAforaves(30*np.arange(len(self.filtered_areatrace[str(label)])),self.filtered_areatrace[str(label)],30*np.arange(len(self.bg_sub_intensity_trace[str(label)])),self.bg_sub_intensity_trace[str(label)])
    def plotIAforaves(self,xdataA,ydataA,xdataI= None,ydataI = None):


        plt.figure()
        if xdataI is not None:

            plt.subplot(211)
            plt.title('Plot of Area (top) and Average Inner Intensity (below) \n against Frame Number for a Randomly Chosen Vesicle')
            plt.ylabel('Area /Pixels')
            plt.yticks([int(np.min(ydataA)),int(0.5*(np.max(ydataA)-np.min(ydataA))),int(np.max(ydataA))])
            plt.xticks(np.array([0,int(len(ydataI)),len(ydataI)])*30)

            plt.plot(xdataA,ydataA)

            plt.subplot(212)
            plt.ylabel('Intensity/arbitrary units')
            plt.yticks([int(np.min(ydataI)),int(0.5*(np.max(ydataI)-np.min(ydataI))),int(np.max(ydataI))])
            plt.xticks(np.array([0,int(len(ydataI)/2),len(ydataI)])*30)
            plt.xlabel('Time since T0/s')
            plt.plot(xdataI,ydataI)

            plt.tight_layout()

        else:
            plt.subplot(111)
            plt.plot(xdataA,ydataA)

        plt.show()
    def viewstream(self,label,tmax,video,t0= None,annotations = True):
        if not t0:
            t0 = self.t0frameNo

        self.ls = livestream(qnd,video[t0:tmax],annotations_on = annotations,annotate_coords = self.centres[str(label)])


    def subtract_background(self,sigma = 3):

        for key in self.firstintensitytrace.keys():

            try:
                self.bg_sub_intensity_trace[key] = np.array(self.intensitytrace[key])-self.bgintens
                self.second_bg_si_trace[key] = np.array(self.secondintensitytrace[key]) -self.bgintens

                self.filtered_intensity_trace[key] = smooth(self.bg_sub_intensity_trace[key],sigma)
                max_data = np.max(self.filtered_intensity_trace[key])

                self.filtered_intensity_trace[key] = self.filtered_intensity_trace[key]/max_data
            except KeyError :
                continue

            
            self.bg_sub_firstintensity_trace[key] = np.array(self.firstintensitytrace[key]) -self.bgintens
            initial_I = self.bg_sub_firstintensity_trace[key][0]
            initial_I = np.array([initial_I])
            
            self.second_bg_fsi_trace[key] = np.array(self.firstsecondintensitytrace[key]) -self.bgintens

            #
            self.filtered_first_intensity_trace[key] = smooth(self.bg_sub_firstintensity_trace[key],sigma)
            self.filtered_first_intensity_trace[key] = smooth(self.firstintensitytrace[key],sigma)
            
            max_data = np.max(self.filtered_first_intensity_trace[key])

        
            self.filtered_first_intensity_trace[key] = self.filtered_first_intensity_trace[key]/max_data
            self.filtered_first_intensity_trace[key] = np.concatenate((initial_I,self.filtered_first_intensity_trace[key]))
            
            print('This is the initial Area, ' ,self.filtered_first_intensity_trace[key][0])


    def heat_data_generator(self):

        #first make sure heat data is array

        self.heat_data = np.array([])

        maxlen = -1
        for key in self.filtered_intensity_trace.keys():
            print(key)
            length = self.filtered_intensity_trace[key].shape[0]
            if length > maxlen:
                maxlen = length

        for key in self.filtered_intensity_trace.keys():

            if self.heat_data.shape[0] == 0:
                self.heat_data = np.concatenate((self.filtered_intensity_trace[key],np.zeros((maxlen-self.filtered_intensity_trace[key].shape[0],))))
            else:

                self.heat_data = np.vstack((self.heat_data,np.concatenate((self.filtered_intensity_trace[key],np.zeros((maxlen-self.filtered_intensity_trace[key].shape[0],))))))
        lengths = []

        for i  in range(0,self.heat_data.shape[0]):
            max_data = np.max(self.heat_data[i,:])
            min_data = np.min(self.heat_data[i,:])



            #make min = zero

            #self.heat_data[i,:] -= min_data

            #divide by the max to make the maximum intensity = 1

            self.heat_data[i,:] = self.heat_data[i,:]/(max_data)

            array = self.heat_data[i,:]

            lengths.append(array[array > 0.5].shape[0])

        lengths = np.array(lengths)


        self.heat_data_ordered = self.heat_data[np.argsort(lengths)]

        self.times = 30*np.arange(self.heat_data_ordered.shape[1])

    def get_heat_plot(self):

        self.HPG.generate(self.times,self.heat_data_ordered,'Plot of the fluorescent intensities of trapped vesicles\n against time after being subject to the drug',self.heat_data_ordered.shape[0])



if __name__ == '__main__':
    '''
    A = Analyser('/Users/MarcusF/Desktop/TrapAnalysis/260522_CecB-  PCPG vesicles- after flushing_1_MMStack_Pos6.ome.tif')
    A.load_frames()
    A.get_traps()
    A.get_clips_alt()
    A.sett0frame(400)
    A.classify_clips()
    A.analyse_frames(965)
    A.extract_background(965)
    A.subtract_background()

    A.visualise_box_contents(11)
    app = QtWidgets.QApplication(sys.argv)
    A.viewstream(11,965,A.vesiclelife)

    A.heat_data_generator()
    A.get_heat_plot()

    A.plotnow(11)
    app.exec_()
    '''
