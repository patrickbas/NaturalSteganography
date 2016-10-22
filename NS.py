# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 17:34:35 2016

@author: bas
"""


import numpy as np
from scipy.special import erf
from scipy.stats import entropy
from scipy import signal
import sys
from PIL import Image
import os
import multiprocessing
from multiprocessing import Pool
import h5py
import glob
#import signal

# Global variables declaration
a_m= 1.0/4
b_l= 1.0/8
b_u= 1.0/8
b_r= 1.0/8
b_d= 1.0/8
c_ul= 1.0/16
c_ur= 1.0/16
c_dr= 1.0/16
c_dl= 1.0/16
mat_filter = np.array([[c_dr,b_d,c_dl] , [b_r,a_m,b_l] , [c_ur,b_u,c_ul]])

#-----------
# Functions
#-----------

#----------------------------------
# Function on images (input cover)
#----------------------------------

# Gamma correction

def gamma_correction_16b(im_cover,gamma):
    if gamma == 1.0:
        return im_cover
    else:
        nb_bits = 16
        lvl_max = float(2**nb_bits-1)
        im_cover_g = lvl_max*(np.power(im_cover/lvl_max,1.0/gamma))
        return im_cover_g

# Subsampling
def sub2_16b(im_cover):
    im_cover_rescale = im_cover[::2,::2]
    return im_cover_rescale

# Box downsampling
def box_down2_16b(im_cover):
    im_cover_rescale = (im_cover[::2,::2]+im_cover[1::2,::2]+im_cover[::2,1::2]+im_cover[1::2,1::2])/4
    return im_cover_rescale

# Tent downsampling

def tent_down2_16b(im_cover):
    im_cover_conv = signal.convolve2d(im_cover,mat_filter,mode='same')
    im_cover_s = im_cover_conv[1::2,1::2]
    return im_cover_s

# Function on the stego signal (input a and b)
#-------------------------------

# Gamma correction (quantization or sampling)

# Subsampling (quantization or sampling)

# Box downsampling (quantization or sampling)

# Tent downsampling (quantization or sampling)


# Diverse functions
#-------------------

# Function to compute entropy and the payload

def entropyMat(p_mat): # p_mat is a h w, rg multidimensional array , rg beeing the number of bins of the pmf
    h,w,rg = p_mat.shape
    rate_map = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            rate_map[i,j]=entropy(p_mat[i,j,:],base=2)
    return np.sum(rate_map)/float(h*w)        

# 16bit to 8 bits quantization

def quant_16b_8b(im_cover_16b,Delta):    # save in 8 bits PGM
    im_cover_8b = np.round((im_cover_16b+1.0)/Delta)
    return im_cover_8b

# Function that modifies the pdf to handle clipping

def simulate_clipping(image_mat,p_map,pbb):
    h,w,rg = p_map.shape
    # <0
    rg = (rg-1)/2
    for p in range(rg):
        p_map[(image_mat+p-rg)<0,p+1]+=p_map[(image_mat+p-rg)<0,p]
        p_map[(image_mat+p-rg)<0,p]=0
    # >255
    for p in range(rg):
        p_map[(image_mat+rg-p)>2**pbb-1,2*rg-1-p]+=p_map[(image_mat+rg-p)>2**pbb-1,2*rg-p]
        p_map[(image_mat+rg-p)>2**pbb-1,2*rg-p]=0
    return p_map
    

# Function that computes the PMF from the (Gaussian) distribution after gamma correction, no gamma correction eq Gamma = 1

def p_map_quant_gaussian_gamma( im_raw , im_raw_d, a, b, Delta , gamma=1):
    h,w = im_raw.shape
    rg_p = 10
    bpp = 16
    v_max = 2**bpp-1
    im_raw
    if gamma !=1:
        var = (a*im_raw+b)*(np.power(im_raw/v_max,-1+1.0/gamma)/gamma)**2
    else:
        var = (a*im_raw+b)
    p_map = np.zeros((h,w,2*rg_p+1))
#    mu_g = v_max*np.power(im_raw/v_max,1.0/gamma)
#    im_raw_g = gamma_correction_16b(im_raw,gamma)

    im_8b = quant_16b_8b(im_raw_d,Delta)
    for p in range(-rg_p,rg_p+1):        
        p1 = p + im_8b 
        p_map[:,:,p+rg_p] = 0.5*( erf(((p1+1.0/2)*Delta-im_raw_d)/(np.sqrt(2*var+sys.float_info.epsilon))) \
        - erf(((p1-1.0/2)*Delta-im_raw_d)/(np.sqrt(2*var+sys.float_info.epsilon))))

    proba_one = np.zeros(2*rg_p+1) 
    proba_one[rg_p] = 1 # wet pixels
    
    # Clipping
    p_map[im_raw_d==v_max,:] = proba_one 
    p_map[im_raw_d==0,:] = proba_one 

    # Don't touch the footstep
    val_min = np.min(im_8b)
    p_map[im_8b==val_min,:] = proba_one 
    
    #Normalisation
    p_map = p_map/np.rollaxis(np.tile(np.sum(p_map,axis=2),(2*rg_p+1,1,1)),0,3)   
    p_map = simulate_clipping(im_8b,p_map,8)

    return p_map
    
# Function to sample from a PMF
def sample_stego_signal(p_map):
    h,w,nb_val = p_map.shape
    sens_noise_8b = np.zeros((h,w)) 
    nb_val = p_map.shape[2]
    rg_val = range(nb_val)
    for i in range(h):
        for j in range(w):
            sens_noise_8b[i,j] = np.random.choice(rg_val, size=1, p=p_map[i,j,:]) #rv_discrete(values=(rg_val,p_map[i,j,:])).rvs(size=1)
    
    sens_noise_8b = sens_noise_8b - nb_val/2
    return sens_noise_8b

# Function to sample for tent downscaling
def sample_stego_signal_tent(im_raw,a,b):
    h_o,w_o = im_raw.shape
    h = h_o/2
    w = w_o/2
    im_var = im_raw*a+b
    
    # Convolutions for grid 1
    mat_filter_2 = mat_filter**2
    conv_arrays_g1 = np.zeros((3,3,9))
    conv_im_g1 = np.zeros((h_o,w_o,9))
    for i in range(3):
        for j in range(3):
            conv_arrays_g1[:,:,i+j*3]=mat_filter_2
            conv_im_g1[:,:,i+j*3] = signal.convolve2d(im_var,conv_arrays_g1[:,:,i+j*3],mode='same')
            mat_filter_2[2-i,2-j]=0

    # Convolutions for grid 2
    mat_filter_2 = mat_filter**2
    mat_filter_2[0,0]=0
    mat_filter_2[1,0]=0
    mat_filter_2[2,0]=0
    mat_filter_2[0,2]=0
    mat_filter_2[1,2]=0
    mat_filter_2[2,2]=0

    conv_arrays_g2 = np.zeros((3,3,3))
    conv_im_g2 = np.zeros((h_o,w_o,3))
    for i in range(3):
        conv_arrays_g2[:,:,i]=mat_filter_2
        conv_im_g2[:,:,i] = signal.convolve2d(im_var,conv_arrays_g2[:,:,i],mode='same')
        mat_filter_2[2-i,1]=0


    # Convolutions for grid 3
    mat_filter_2 = mat_filter**2
    mat_filter_2[0,0]=0
    mat_filter_2[0,1]=0
    mat_filter_2[0,2]=0
    mat_filter_2[2,0]=0
    mat_filter_2[2,1]=0
    mat_filter_2[2,2]=0

    conv_arrays_g3 = np.zeros((3,3,3))
    conv_im_g3 = np.zeros((h_o,w_o,3))
    for i in range(3):
        conv_arrays_g3[:,:,i]=mat_filter_2
        conv_im_g3[:,:,i] = signal.convolve2d(im_var,conv_arrays_g3[:,:,i],mode='same')
        mat_filter_2[1,2-i]=0


    # Convolutions for grid 4 (stupid)
    mat_filter_2 = mat_filter**2
    mat_filter_2[0,0]=0
    mat_filter_2[0,1]=0
    mat_filter_2[0,2]=0
    mat_filter_2[2,0]=0
    mat_filter_2[2,1]=0
    mat_filter_2[2,2]=0
    mat_filter_2[1,0]=0
    mat_filter_2[1,2]=0

    conv_arrays_g4 = np.zeros((3,3,1))
    conv_im_g4 = np.zeros((2*h,2*w,1))
    conv_arrays_g4[:,:,0]=mat_filter_2
    conv_im_g4[:,:,0] = signal.convolve2d(im_var,conv_arrays_g4[:,:,0],mode='same')


    # Algo:
    # initialisations
    # Matrice of Sensor noise (aka photosite noise)
    S = np.zeros((h_o,w_o))
    # Matrice of pixel noise
    S_s = np.zeros((h,w))
    
    # Step 1:
    # Draw the pixel noise on lattice L1
    S_s[::2,::2] = np.random.randn(h/2,w/2)*np.sqrt(conv_im_g1[1::4,1::4,0])
    # Draw the sensor noise conditionaly to the pixel noise and the previously drawn sensor noise
    for i in range(3):        
        for j in range(3):        
            #print 'i,j:' , i,j
            var_sub=    im_var[i::4,j::4]
            not_drawn = conv_im_g1[1::4,1::4,i+j*3]
            conv_sens_noise = signal.convolve2d(S,mat_filter,mode='same')
            mean = (S_s[::2,::2] - conv_sens_noise[1::4,1::4]) * mat_filter[i,j]*var_sub/not_drawn
            var = var_sub - (mat_filter[i,j]*var_sub)**2/not_drawn
            #print var[3:6,3:6]
            S[i::4,j::4] =  mean + np.random.randn(h/2,w/2)*np.sqrt(np.abs(var))
            
    # Step 2:
    # Draw the pixel noise on lattice L2
    conv_sens_noise = signal.convolve2d(S,mat_filter,mode='same')
    S_s[::2,1::2] = conv_sens_noise[1::4,3::4]+ np.random.randn(h/2,w/2)*np.sqrt(conv_im_g2[1::4,3::4,0])
    # Draw the sensor noise conditionaly to the pixel noise and the previously drawn sensor noise
    for i in range(3):        
        #print 'i:' , i
        var_sub=    im_var[i::4,3::4]
        not_drawn = conv_im_g2[1::4,3::4,i]
        conv_sens_noise = signal.convolve2d(S,mat_filter,mode='same')
        mean = (S_s[::2,1::2] - conv_sens_noise[1::4,3::4]) * mat_filter[i,1]*var_sub/not_drawn
        var = var_sub - (mat_filter[i,1]*var_sub)**2/not_drawn
        #print var[3:6,3:6]
        S[i::4,3::4] =  mean + np.random.randn(h/2,w/2)*np.sqrt(np.abs(var))
            
    # Step 3:
    # Draw the pixel noise on lattice L3
    conv_sens_noise = signal.convolve2d(S,mat_filter,mode='same')
    S_s[1::2,::2] = conv_sens_noise[3::4,1::4]+ np.random.randn(h/2,w/2)*np.sqrt(conv_im_g3[3::4,1::4,0])
    # Draw the sensor noise conditionaly to the pixel noise and the previously drawn sensor noise
    for i in range(3):        
        #print 'i:' , i
        var_sub=    im_var[3::4,i::4]
        not_drawn = conv_im_g3[3::4,1::4,i]
        conv_sens_noise = signal.convolve2d(S,mat_filter,mode='same')
        mean = (S_s[1::2,::2] - conv_sens_noise[3::4,1::4]) * mat_filter[1,i]*var_sub/not_drawn
        var = var_sub - (mat_filter[1,i]*var_sub)**2/not_drawn
        #print var[3:6,3:6]
        S[3::4,i::4] =  mean + np.random.randn(h/2,w/2)*np.sqrt(np.abs(var))

    # Step 4:
    # Draw the pixel noise on lattice L4
    conv_sens_noise = signal.convolve2d(S,mat_filter,mode='same')
    S_s[1::2,1::2] = conv_sens_noise[3::4,3::4]+ np.random.randn(h/2,w/2)*np.sqrt(conv_im_g4[3::4,3::4,0])

    var_sub= im_var[3::4,3::4]
    not_drawn = conv_im_g4[3::4,3::4,0]
    conv_sens_noise = signal.convolve2d(S,mat_filter,mode='same')
    mean = (S_s[1::2,1::2] - conv_sens_noise[3::4,3::4]) * mat_filter[1,1]*var_sub/not_drawn
    var = var_sub - (mat_filter[1,1]*var_sub)**2/not_drawn
    S[3::4,3::4] =  mean + np.random.randn(h/2,w/2)*np.sqrt(np.abs(var))

    return np.round(S_s)

# Outputs
#---------

# Generate Stego and Cover
# Generate only Cover
# Generate only Stego

# Requirements
#--------------

# In order to run programs in parralel
# Cover: directory named with the developping process, ISO Setting
# Stego: directory name with de velolopping process, ISO Setting, a and b

def main_NS_im(image_name):
    nb_bits = main_NS_im.nb_bits
   
    a = main_NS_im.a * 2**nb_bits
    b = main_NS_im.b * 2**(2*nb_bits)
    fast = main_NS_im.fast
    gamma = main_NS_im.gamma 
    out_flag = main_NS_im.out_flag
    dev = main_NS_im.dev
    pil_cover = Image.open(image_name)
    im_cover = np.asarray(pil_cover).astype(float)
    h,w = im_cover.shape
    Delta = 256.0
    new_name = os.path.basename(image_name)

    if dev == 'Gamma':
        # Gamma transform:
        im_cover_d = gamma_correction_16b(im_cover,gamma)
    elif dev == 'Box':
        im_cover_d = box_down2_16b(im_cover)
        im_cover = im_cover_d
        a = a/4
        b = b/4
        w = w/2
        h = h/2
    elif dev == 'Sub':
        im_cover_d = sub2_16b(im_cover)
        im_cover = im_cover_d
        w = w/2
        h = h/2
    elif dev == 'Tent' or dev == 'Tent2' :
        im_cover_d = tent_down2_16b(im_cover)
        w = w/2
        h = h/2

    # save in 8 bits PGM
    im_cover_8b = quant_16b_8b(im_cover_d,Delta) 
    if fast==0: # sample in the developed domain
        if dev == 'Gamma' or dev == 'Box' or dev == 'Sub': 
            p_map = p_map_quant_gaussian_gamma(im_cover , im_cover_d , a, b, Delta, gamma=gamma)
            sens_noise_8b = sample_stego_signal(p_map)
        im_stego_8b = im_cover_8b + sens_noise_8b
        print 'sum: ' , np.sum(np.abs(sens_noise_8b))
        
    else: # sample then develop (as a baseline)
        if dev == 'Gamma' or dev == 'Box' or dev == 'Sub': 
            sens_noise = np.random.randn(h,w)*np.sqrt(im_cover*a+b)
            im_stego = im_cover + sens_noise
            lvl_max = 2**nb_bits - 1
            # Do not touch saturated pixels
            im_stego[im_cover==0]=0
            im_stego[im_cover==lvl_max]=lvl_max

        elif dev == 'Tent':
            sens_noise = tent_down2_16b(np.random.randn(h*2,w*2)*np.sqrt(im_cover*a+b))
            im_stego = im_cover_d + sens_noise
            lvl_max = 2**nb_bits - 1
            # Do not touch saturated pixels
            im_stego[im_cover_d==0]=0
            im_stego[im_cover_d==lvl_max]=lvl_max
        elif dev == 'Tent2':
            sens_noise = sample_stego_signal_tent(im_cover,a,b)
            im_stego = im_cover_d + sens_noise
            lvl_max = 2**nb_bits - 1
            # Do not touch saturated pixels
            im_stego[im_cover_d==0]=0
            im_stego[im_cover_d==lvl_max]=lvl_max

        # Clipping
        im_stego[im_stego<=0]=0
        im_stego[im_stego>=lvl_max]=lvl_max

        im_stego_d = gamma_correction_16b(im_stego,gamma)
        im_stego_8b = quant_16b_8b(im_stego_d,Delta) 
        print 'sum: ' , np.sum(np.abs(im_stego_8b-im_cover_8b))

        # Wet pixels
        val_min = np.min(im_cover_8b)
        im_stego_8b[im_cover_8b==val_min]=val_min


    # Generate the appropriate images
    if out_flag == 1 or out_flag == 3: # Save Cover
        im_cover_8b = im_cover_8b.astype(dtype=np.uint8)
        im_cover_pgm = Image.fromarray(im_cover_8b)
        im_cover_pgm.save(main_NS_im.cover_8_dir+new_name)

    if out_flag == 2 or out_flag == 3: # Save Stego
        im_stego_8b = im_stego_8b.astype(dtype=np.uint8)
        im_stego_pgm = Image.fromarray(im_stego_8b)
        im_stego_pgm.save(main_NS_im.stego_8_dir+new_name)

    # Compute the embedding rate    
    if fast ==0:
        emb_rate = entropyMat(p_map)
    else :
        emb_rate = 0      
        
    print new_name + " embedded"
    return (emb_rate)
    
def main_NS(in_dir,a,b,ISO_set,dev,gamma=1,out_flag=3,debug=0,fast=0):     

        
    main_NS_im.nb_bits = 16

    main_NS_im.a = a 
    main_NS_im.b = b
    main_NS_im.gamma = gamma
    main_NS_im.out_flag = out_flag
    main_NS_im.fast = fast
    main_NS_im.nb_bits = 16
    main_NS_im.dev = dev


    main_NS_im.in_dir = in_dir
    main_NS_im.cover_16_dir = main_NS_im.in_dir+'Cover_'+ISO_set+'_512_16b/'
    main_NS_im.cover_8_dir = main_NS_im.in_dir+'Cover_'+ISO_set+\
    '_512_8b'+'_'+str(a)+'_'+str(b)+'_'+dev+'_'+str(fast)+'/'
    main_NS_im.stego_8_dir = main_NS_im.in_dir+'Stego_'+ISO_set+\
    '_512_8b'+'_'+str(a)+'_'+str(b)+'_'+dev+'_'+str(fast)+'/'
    
    if not os.path.exists(main_NS_im.cover_8_dir):
        os.makedirs(main_NS_im.cover_8_dir)

    if not os.path.exists(main_NS_im.stego_8_dir):
        os.makedirs(main_NS_im.stego_8_dir)

    list_im = sorted(glob.glob(main_NS_im.cover_16_dir+'*.pgm'))

    if(debug==0):
        nbCores = multiprocessing.cpu_count()
        pool = Pool(nbCores)
        list_e_r = pool.map(main_NS_im, list_im)
        pool.close()
        pool.join()
        fileOut = h5py.File(main_NS_im.cover_8_dir+'e_r.hdf5','w')
        fileOut.create_dataset('dataset_1', data=list_e_r + list_im)
        fileOut.close()    
        print 'average embedding rate:' , np.mean(list_e_r), ' bpp' 
    else:
        e_r = main_NS_im(list_im[0])
        print 'embedding rate:' , e_r , ' bpp' 
        
    return(1)
    



if __name__ == "__main__":

    path_base = ''
    path_dir = ''
    a = 2.5*1e-5
    b = 8.0*1e-7
    ISO_set =  '1000'
    gam = 1
    out_f = 1
    #dev = "Gamma"
    #dev = "Box"
    #dev = "Sub"
    dev = "Tent2"
    main_NS(path_base+path_dir,a,b,ISO_set,dev,debug=1,fast=1)
    
