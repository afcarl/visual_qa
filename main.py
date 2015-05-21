from __future__ import division
import argparse
import cPickle
import lasagne
import numpy as np
import os
import pyprind
import re
import scipy
import sys
import theano
import theano.tensor as T
import time
from collections import Counter
from lasagne.layers import cuda_convnet
from nltk.corpus import stopwords
from sklearn.cross_validation import *
from sklearn.decomposition import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from theano.ifelse import ifelse
from theano.printing import Print as pp
from calculate_wups import dirac_measure, fuzzy_set_membership_measure, items2list, score_it, wup_measure
from utils import *

USE_SINGLE_ANSWER = True
BATCH_SIZE = 128
FEATURES_FILE = 'vgg/features.pkl'
STOPWORDS = stopwords.words('english')

TRAIN_FILE = 'data/daquar37/qa.37.raw.train.txt'
TEST_FILE = 'data/daquar37/qa.37.raw.test.txt'
IMG_LIST = [os.path.expanduser(f.strip()) for f in open('vgg/files.txt')]

W2V_FILE = 'embeddings/word2vec/GoogleNews-vectors-negative300.bin'
GLOVE_FILE = 'embeddings/glove/glove.840B.300d.txt'

TYPOS = {
    'clockes': 'clocks',
    'toyhouse': 'toy house',
    'benhind': 'behind',
    'squer': 'square',
    'liquod': 'liquid',
    'tshirtsg': 'tshirts',
    'firepalce': 'fireplace',
    'corck': 'cork',
    'viisble': 'visible',
    'cupbaord': 'cupboard',
    'eimage999': 'image999',
    'beneatht': 'beneath',
    'inbetweeng': 'in between',
    'airconditionerg': 'air conditioner',
    'objests': 'objects',
    'objest': 'object'
}

MAX_LEN = 0

def reduce_along_dim(img, dim, weights, indicies): 
    '''
    Perform bilinear interpolation given along the image dimension dim
    -weights are the kernel weights 
    -indicies are the crossponding indicies location
    return img resize along dimension dim
    '''
    other_dim = abs(dim-1)       
    if other_dim == 0:  #resizing image width
        weights  = np.tile(weights[np.newaxis,:,:,np.newaxis],(img.shape[other_dim],1,1,3))
        out_img = img[:,indicies,:]*weights
        out_img = np.sum(out_img,axis=2)
    else:   # resize image height     
        weights  = np.tile(weights[:,:,np.newaxis,np.newaxis],(1,1,img.shape[other_dim],3))
        out_img = img[indicies,:,:]*weights
        out_img = np.sum(out_img,axis=1)
        
    return out_img

            
def cubic_spline(x):
    '''
    Compute the kernel weights 
    See Keys, "Cubic Convolution Interpolation for Digital Image
    Processing," IEEE Transactions on Acoustics, Speech, and Signal
    Processing, Vol. ASSP-29, No. 6, December 1981, p. 1155.
    '''
    absx   = np.abs(x)
    absx2  = absx**2
    absx3  = absx**3 
    kernel_weight = (1.5*absx3 - 2.5*absx2 + 1) * (absx<=1) + (-0.5*absx3 + 2.5* absx2 - 4*absx + 2) * ((1<absx) & (absx<=2))
    return kernel_weight
    
def contribution(in_dim_len, out_dim_len, scale ):
    '''
    Compute the weights and indicies of the pixels involved in the cubic interpolation along each dimension.
    
    output:
    weights a list of size 2 (one set of weights for each dimension). Each item is of size OUT_DIM_LEN*Kernel_Width
    indicies a list of size 2(one set of pixel indicies for each dimension) Each item is of size OUT_DIM_LEN*kernel_width
    
    note that if the entire column weights is zero, it gets deleted since those pixels don't contribute to anything
    '''
    kernel_width = 4
    if scale < 1:
        kernel_width =  4 / scale
        
    x_out = np.array(range(1,out_dim_len+1))  
    #project to the input space dimension
    u = x_out/scale + 0.5*(1-1/scale)
    
    #position of the left most pixel in each calculation
    l = np.floor( u - kernel_width/2)
  
    #maxium number of pixels in each computation
    p = int(np.ceil(kernel_width) + 2)
    
    indicies = np.zeros((l.shape[0],p), dtype = int)
    indicies[:,0] = l
      
    for i in range(1,p):
        indicies[:,i] = indicies[:,i-1]+1
    
    #compute the weights of the vectors
    u = u.reshape((u.shape[0],1))
    u = np.repeat(u,p,axis=1)
    
    if scale < 1:
        weights = scale*cubic_spline(scale*(indicies-u ))
    else:
        weights = cubic_spline((indicies-u))
         
    weights_sums = np.sum(weights,1)
    weights = weights/ weights_sums[:, np.newaxis] 
    
    indicies = indicies - 1    
    indicies[indicies<0] = 0                     
    indicies[indicies>in_dim_len-1] = in_dim_len-1 #clamping the indicies at the ends
    
    valid_cols = np.all( weights==0, axis = 0 ) == False #find columns that are not all zeros
    
    indicies  = indicies[:,valid_cols]           
    weights    = weights[:,valid_cols]
    
    return weights, indicies

def imresize(img, cropped_width, cropped_height):
    '''
    Function implementing matlab's imresize functionality default behaviour
    Cubic spline interpolation with antialiasing correction when scaling down the image.
    
    '''
    
    
    width_scale  = float(cropped_width)  / img.shape[1]
    height_scale = float(cropped_height) / img.shape[0] 
    
    if len(img.shape) == 2: #Gray Scale Case
        img = np.tile(img[:,:,np.newaxis], (1,1,3)) #Broadcast 
    
    order   = np.argsort([height_scale, width_scale])
    scale   = [height_scale, width_scale]
    out_dim = [cropped_height, cropped_width] 
    
    
    weights  = [0,0]
    indicies = [0,0]
    
    for i in range(0, 2):
        weights[i], indicies[i] = contribution(img.shape[ i ],out_dim[i], scale[i])
    
    for i in range(0, len(order)):
        img = reduce_along_dim(img, order[i], weights[order[i]], indicies[order[i]])
        
    return img

def preprocess_image(img):
    '''
    Preprocess an input image before processing by the caffe module.
    
    
    Preprocessing include:
    -----------------------
    1- Converting image to single precision data type
    2- Resizing the input image to cropped_dimensions used in extract_features() matlab script
    3- Reorder color Channel, RGB->BGR
    4- Convert color scale from 0-1 to 0-255 range (actually because image type is a float the 
        actual range could be negative or >255 during the cubic spline interpolation for image resize.
    5- Subtract the VGG dataset mean.
    6- Reorder the image to standard caffe input dimension order ( 3xHxW) 
    '''
    img      = img.astype(np.float32)
    img      = imresize(img,224,224) #cropping the image
    img      = img[:,:,[2,1,0]] #RGB-BGR
    img      = img*255
    
    mean = np.array([103.939, 116.779, 123.68]) #mean of the vgg 
    
    for i in range(0,3):
        img[:,:,i] = img[:,:,i] - mean[i] #subtracting the mean
    img = np.transpose(img, [2,0,1])
    return img #HxWx3

def clean(s):
    if re.match('image[0-9]+', s):
        return 'image'
    if s in TYPOS:
        return TYPOS[s]
    return s

def escapeNumber(line):
    line = re.sub('^21$', 'twenty_one', line)
    line = re.sub('^22$', 'twenty_two', line)
    line = re.sub('^23$', 'twenty_three', line)
    line = re.sub('^24$', 'twenty_four', line)
    line = re.sub('^25$', 'twenty_five', line)
    line = re.sub('^26$', 'twenty_six', line)
    line = re.sub('^27$', 'twenty_seven', line)
    line = re.sub('^28$', 'twenty_eight', line)
    line = re.sub('^29$', 'twenty_nine', line)
    line = re.sub('^30$', 'thirty', line)
    line = re.sub('^11$', 'eleven', line)
    line = re.sub('^12$', 'twelve', line)
    line = re.sub('^13$', 'thirteen', line)
    line = re.sub('^14$', 'fourteen', line)
    line = re.sub('^15$', 'fifteen', line)
    line = re.sub('^16$', 'sixteen', line)
    line = re.sub('^17$', 'seventeen', line)
    line = re.sub('^18$', 'eighteen', line)
    line = re.sub('^19$', 'nineteen', line)
    line = re.sub('^20$', 'twenty', line)
    line = re.sub('^10$', 'ten', line)
    line = re.sub('^0$', 'zero', line)
    line = re.sub('^1$', 'one', line)
    line = re.sub('^2$', 'two', line)
    line = re.sub('^3$', 'three', line)
    line = re.sub('^4$', 'four', line)
    line = re.sub('^5$', 'five', line)
    line = re.sub('^6$', 'six', line)
    line = re.sub('^7$', 'seven', line)
    line = re.sub('^8$', 'eight', line)
    line = re.sub('^9$', 'nine', line)
    return line

def extract_qa(lines):
    global MAX_LEN
    questions = []
    answers = []
    imgIds = []
    lineMax = 0
    for i in range(0, len(lines) // 2):
        n = i * 2
        if ',' in lines[n + 1]:
            # No multiple words answer for now.
            continue
        match = re.search('image(\d+)', lines[n])
        number = int((re.search('\d+', match.group())).group())
        line = lines[n]
        line = re.sub(' in the image(\d+)( \?\s)?', '' , line)
        line = re.sub(' in this image(\d+)( \?\s)?', '' , line)
        line = re.sub(' on the image(\d+)( \?\s)?', '' , line)
        line = re.sub(' of the image(\d+)( \?\s)?', '' , line)
        line = re.sub(' in image(\d+)( \?\s)?', '' , line)
        line = re.sub(' image(\d+)( \?\s)?', '' , line)
        words = [clean(w) for w in line.split()]
        line = ' '.join(words)
        MAX_LEN = max(MAX_LEN, len(words))
        questions.append(line)
        answer = escapeNumber(re.sub('\s$', '', lines[n + 1]))
        answers.append(answer)
        # Important! Here image id is 0-based.
        imgIds.append(number - 1)
    return (questions, answers, imgIds)

def data_split(data, imgids, split):
    td = []
    vd = []
    for (d, i) in zip(data, imgids):
        if split[i] == 0:
            vd.append(d)
        else:
            td.append(d)
    return (td, vd)

def train_valid_split(imgids):
    split = {}
    for i in imgids:
        split[i] = 1
    count = 0
    for i in split.keys():
        if count < len(split) / 10:
            split[i] = 0
        else:
            break
        count += 1
    return split

def load_file(fname):
    global MAX_LEN
    L = [line.strip() for line in open(fname)]
    return extract_qa(L)

    Q = L[0::2]
    A = [y.split(', ') for y in L[1::2]]
    I = []
    for i,x in enumerate(Q):
        words = x.split()        
        img_id = int(re.sub('[^0-9]', '', words[-2]))-1        
        I.append(img_id)
        words = words[:-4]
        words = [clean(w) for w in words]
        Q[i] = ' '.join(words)
        MAX_LEN = max(len(Q[i].split()), MAX_LEN)
    if USE_SINGLE_ANSWER:
        Q, A, I = [list(t) for t in zip(*filter(lambda z: len(z[1]) == 1, zip(Q, A, I)))]
        A = [z[0] for z in A]
    return Q, A, I

X_train_val, Y_train_val_raw, I_train_val = load_file(TRAIN_FILE)
X_test, Y_test_raw, I_test = load_file(TEST_FILE)

split = train_valid_split(I_train_val)
X_train, X_val = data_split(X_train_val, I_train_val, split)
Y_train_raw, Y_val_raw = data_split(Y_train_val_raw, I_train_val, split)
I_train, I_val = data_split(I_train_val, I_train_val, split)

"""
X_train, X_val, Y_train_raw, Y_val_raw, I_train, I_val = train_test_split(X_train_val,
                                                                          Y_train_val_raw,
                                                                          I_train_val,
                                                                          test_size=512,
                                                                          random_state=42)
"""                                                                          

dataset = { 'train': { 'q': X_train, 'y': Y_train_raw, 'img_idxs': I_train },
            'val': { 'q': X_val, 'y': Y_val_raw, 'img_idxs': I_val },
            'test': { 'q': X_test, 'y': Y_test_raw, 'img_idxs': I_test } }

for data in dataset.keys():
    for key in dataset[data].keys():
        print data, key, len(dataset[data][key])
        dataset[data][key] = pad_to_batch_size(dataset[data][key], BATCH_SIZE)
        print data, key, len(dataset[data][key])

if USE_SINGLE_ANSWER:
    mlb = LabelBinarizer()
else:
    mlb = MultiLabelBinarizer()
mlb.fit(Y_train_val_raw + Y_test_raw)

for data in dataset.keys():
    dataset[data]['y'] = mlb.transform(dataset[data]['y'])
    
print "MAX_LEN: ", MAX_LEN
print "ANSWERS: ", len(mlb.classes_)

def s_tanh(x):
    return 1.7159 * T.tanh(x*2.0/3.0)

class LSTMLayer(lasagne.layers.Layer):
    '''
    A long short-term memory (LSTM) layer.  Includes "peephole connections" and
    forget gate.  Based on the definition in [#graves2014generating]_, which is
    the current common definition. Gate names are taken from [#zaremba2014],
    figure 1.

    :references:
        .. [#graves2014generating] Alex Graves, "Generating Sequences With
            Recurrent Neural Networks"
        .. [#zaremba2014] Wojciech Zaremba et al.,  "Recurrent neural network
           regularization"
    '''
    def __init__(self, input_layer, num_units,
                 W_in_to_ingate=lasagne.init.Normal(0.1),
                 W_hid_to_ingate=lasagne.init.Normal(0.1),
                 W_cell_to_ingate=lasagne.init.Normal(0.1),
                 b_ingate=lasagne.init.Normal(0.1),
                 nonlinearity_ingate=lasagne.nonlinearities.sigmoid,
                 W_in_to_forgetgate=lasagne.init.Normal(0.1),
                 W_hid_to_forgetgate=lasagne.init.Normal(0.1),
                 W_cell_to_forgetgate=lasagne.init.Normal(0.1),
                 b_forgetgate=lasagne.init.Normal(0.1),
                 nonlinearity_forgetgate=lasagne.nonlinearities.sigmoid,
                 W_in_to_cell=lasagne.init.Normal(0.1),
                 W_hid_to_cell=lasagne.init.Normal(0.1),
                 b_cell=lasagne.init.Normal(0.1),
                 nonlinearity_cell=lasagne.nonlinearities.tanh,
                 W_in_to_outgate=lasagne.init.Normal(0.1),
                 W_hid_to_outgate=lasagne.init.Normal(0.1),
                 W_cell_to_outgate=lasagne.init.Normal(0.1),
                 b_outgate=lasagne.init.Normal(0.1),
                 nonlinearity_outgate=lasagne.nonlinearities.sigmoid,
                 nonlinearity_out=lasagne.nonlinearities.tanh,
                 cell_init=lasagne.init.Constant(0.),
                 hid_init=lasagne.init.Constant(0.),
                 W_in_to_imgingate=lasagne.init.Normal(0.1),                 
                 W_in_to_imgforgetgate=lasagne.init.Normal(0.1),
                 W_in_to_imgcell=lasagne.init.Normal(0.1),
                 W_in_to_imgoutgate=lasagne.init.Normal(0.1),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1):
        '''
        Initialize an LSTM layer.  For details on what the parameters mean, see
        (7-11) from [#graves2014generating]_.

        :parameters:
            - input_layer : layers.Layer
                Input to this recurrent layer
            - num_units : int
                Number of hidden units
            - W_in_to_ingate : function or np.ndarray or theano.shared
                :math:`W_{xi}`
            - W_hid_to_ingate : function or np.ndarray or theano.shared
                :math:`W_{hi}`
            - W_cell_to_ingate : function or np.ndarray or theano.shared
                :math:`W_{ci}`
            - b_ingate : function or np.ndarray or theano.shared
                :math:`b_i`
            - nonlinearity_ingate : function
                :math:`\sigma`
            - W_in_to_forgetgate : function or np.ndarray or theano.shared
                :math:`W_{xf}`
            - W_hid_to_forgetgate : function or np.ndarray or theano.shared
                :math:`W_{hf}`
            - W_cell_to_forgetgate : function or np.ndarray or theano.shared
                :math:`W_{cf}`
            - b_forgetgate : function or np.ndarray or theano.shared
                :math:`b_f`
            - nonlinearity_forgetgate : function
                :math:`\sigma`
            - W_in_to_cell : function or np.ndarray or theano.shared
                :math:`W_{ic}`
            - W_hid_to_cell : function or np.ndarray or theano.shared
                :math:`W_{hc}`
            - b_cell : function or np.ndarray or theano.shared
                :math:`b_c`
            - nonlinearity_cell : function or np.ndarray or theano.shared
                :math:`\tanh`
            - W_in_to_outgate : function or np.ndarray or theano.shared
                :math:`W_{io}`
            - W_hid_to_outgate : function or np.ndarray or theano.shared
                :math:`W_{ho}`
            - W_cell_to_outgate : function or np.ndarray or theano.shared
                :math:`W_{co}`
            - b_outgate : function or np.ndarray or theano.shared
                :math:`b_o`
            - nonlinearity_outgate : function
                :math:`\sigma`
            - nonlinearity_out : function or np.ndarray or theano.shared
                :math:`\tanh`
            - cell_init : function or np.ndarray or theano.shared
                :math:`c_0`
            - hid_init : function or np.ndarray or theano.shared
                :math:`h_0`
            - backwards : boolean
                If True, process the sequence backwards and then reverse the
                output again such that the output from the layer is always
                from x_1 to x_n.
            - learn_init : boolean
                If True, initial hidden values are learned
            - peepholes : boolean
                If True, the LSTM uses peephole connections.
                When False, W_cell_to_ingate, W_cell_to_forgetgate and
                W_cell_to_outgate are ignored.
            - gradient_steps : int
                Number of timesteps to include in backpropagated gradient
                If -1, backpropagate through the entire sequence
        '''

        # Initialize parent layer
        super(LSTMLayer, self).__init__(input_layer)

        # For any of the nonlinearities, if None is supplied, use identity
        if nonlinearity_ingate is None:
            self.nonlinearity_ingate = nonlinearities.identity
        else:
            self.nonlinearity_ingate = nonlinearity_ingate

        if nonlinearity_forgetgate is None:
            self.nonlinearity_forgetgate = nonlinearities.identity
        else:
            self.nonlinearity_forgetgate = nonlinearity_forgetgate

        if nonlinearity_cell is None:
            self.nonlinearity_cell = nonlinearities.identity
        else:
            self.nonlinearity_cell = nonlinearity_cell

        if nonlinearity_outgate is None:
            self.nonlinearity_outgate = nonlinearities.identity
        else:
            self.nonlinearity_outgate = nonlinearity_outgate

        if nonlinearity_out is None:
            self.nonlinearity_out = nonlinearities.identity
        else:
            self.nonlinearity_out = nonlinearity_out

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps

        # Input dimensionality is the output dimensionality of the input layer
        (num_batch, _, num_inputs) = self.input_layer.get_output_shape()

        # Initialize parameters using the supplied args
        self.W_in_to_ingate = self.create_param(
            W_in_to_ingate, (num_inputs, num_units), name="W_in_to_ingate")

        self.W_hid_to_ingate = self.create_param(
            W_hid_to_ingate, (num_units, num_units), name="W_hid_to_ingate")

        self.b_ingate = self.create_param(
            b_ingate, (num_units), name="b_ingate")

        self.W_in_to_forgetgate = self.create_param(
            W_in_to_forgetgate, (num_inputs, num_units),
            name="W_in_to_forgetgate")

        self.W_hid_to_forgetgate = self.create_param(
            W_hid_to_forgetgate, (num_units, num_units),
            name="W_hid_to_forgetgate")

        self.b_forgetgate = self.create_param(
            b_forgetgate, (num_units,), name="b_forgetgate")

        self.W_in_to_cell = self.create_param(
            W_in_to_cell, (num_inputs, num_units), name="W_in_to_cell")

        self.W_hid_to_cell = self.create_param(
            W_hid_to_cell, (num_units, num_units), name="W_hid_to_cell")

        self.b_cell = self.create_param(
            b_cell, (num_units,), name="b_cell")

        self.W_in_to_outgate = self.create_param(
            W_in_to_outgate, (num_inputs, num_units), name="W_in_to_outgate")

        self.W_hid_to_outgate = self.create_param(
            W_hid_to_outgate, (num_units, num_units), name="W_hid_to_outgate")

        self.b_outgate = self.create_param(
            b_outgate, (num_units,), name="b_outgate")

        self.W_in_to_imgingate = self.create_param(
            W_in_to_imgingate, (num_inputs, num_units), name="W_in_to_imgingate")
        
        self.W_in_to_imgforgetgate = self.create_param(
            W_in_to_imgforgetgate, (num_inputs, num_units),
            name="W_in_to_imgforgetgate")
        
        self.W_in_to_imgcell = self.create_param(
            W_in_to_imgcell, (num_inputs, num_units), name="W_in_to_imgcell")
        
        self.W_in_to_imgoutgate = self.create_param(
            W_in_to_imgoutgate, (num_inputs, num_units), name="W_in_to_imgoutgate")        

        # Stack input to gate weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        self.W_in_to_gates = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
            self.W_in_to_cell, self.W_in_to_outgate], axis=1)
        
        self.W_in_to_imggates = T.concatenate(
            [self.W_in_to_imgingate, self.W_in_to_imgforgetgate,
            self.W_in_to_imgcell, self.W_in_to_imgoutgate], axis=1)
            
        # Same for hidden to gate weight matrices
        self.W_hid_to_gates = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
            self.W_hid_to_cell, self.W_hid_to_outgate, ], axis=1)

        # Stack gate biases into a (4*num_units) vector
        self.b_gates = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
            self.b_cell, self.b_outgate], axis=0)

        # Initialize peephole (cell to gate) connections.  These are
        # elementwise products with the cell state, so they are represented as
        # vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.create_param(
                W_cell_to_ingate, (num_units), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.create_param(
                W_cell_to_forgetgate, (num_units), name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.create_param(
                W_cell_to_outgate, (num_units), name="W_cell_to_outgate")

        # Setup initial values for the cell and the hidden units
        self.cell_init = self.create_param(
            cell_init, (num_batch, num_units), name="cell_init")
        self.hid_init = self.create_param(
            hid_init, (num_batch, num_units), name="hid_init")

    def get_params(self):
        '''
        Get all parameters of this layer.

        :returns:
            - params : list of theano.shared
                List of all parameters
        '''
        params = self.get_weight_params() + self.get_bias_params()
        if self.peepholes:
            params.extend(self.get_peephole_params())

        if self.learn_init:
            params.extend(self.get_init_params())

        return params

    def get_weight_params(self):
        '''
        Get all weight matrix parameters of this layer

        :returns:
            - weight_params : list of theano.shared
                List of all weight matrix parameters
        '''
        return [self.W_in_to_ingate,
                self.W_hid_to_ingate,
                self.W_in_to_forgetgate,
                self.W_hid_to_forgetgate,
                self.W_in_to_cell,
                self.W_hid_to_cell,
                self.W_in_to_outgate,
                self.W_hid_to_outgate,                
                self.W_in_to_imgingate,
                self.W_in_to_imgforgetgate,
                self.W_in_to_imgcell,
                self.W_in_to_imgoutgate
                ]

    def get_peephole_params(self):
        '''
        Get all peephole connection parameters of this layer.

        :returns:
            - peephole_params : list of theano.shared
                List of all peephole parameters.  If this LSTM layer doesn't
                use peephole connections (peepholes=False), then an empty list
                is returned.
        '''
        if self.peepholes:
            return [self.W_cell_to_ingate,
                    self.W_cell_to_forgetgate,
                    self.W_cell_to_outgate]
        else:
            return []

    def get_init_params(self):
        '''
        Get all initital state parameters of this layer.

        :returns:
            - init_params : list of theano.shared
                List of all initial parameters
        '''
        return [self.hid_init, self.cell_init]

    def get_bias_params(self):
        '''
        Get all bias parameters of this layer.

        :returns:
            - bias_params : list of theano.shared
                List of all bias parameters
        '''
        return [self.b_ingate, self.b_forgetgate,
                self.b_cell, self.b_outgate]

    def get_output_shape_for(self, input_shape):
        '''
        Compute the expected output shape given the input.

        :parameters:
            - input_shape : tuple
                Dimensionality of expected input

        :returns:
            - output_shape : tuple
                Dimensionality of expected outputs given input_shape
        '''
        return (input_shape[0], input_shape[1], self.num_units)

    def get_output_for(self, input, mask=None, e_imgfeats=0, *args, **kwargs):
        '''
        Compute this layer's output function given a symbolic input variable

        :parameters:
            - input : theano.TensorType
                Symbolic input variable
            - mask : theano.TensorType
                Theano variable denoting whether each time step in each
                sequence in the batch is part of the sequence or not.  If None,
                then it assumed that all sequences are of the same length.  If
                not all sequences are of the same length, then it must be
                supplied as a matrix of shape (n_batch, n_time_steps) where
                `mask[i, j] = 1` when `j <= (length of sequence i)` and
                `mask[i, j] = 0` when `j > (length of sequence i)`.

        :returns:
            - layer_output : theano.TensorType
                Symbolic output variable
        '''
        # Treat all layers after the first as flattened feature dimensions
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)

        # Because the input is given for all time steps, we can precompute
        # the inputs to the gates before scanning.
        # input is dimshuffled to (n_time_steps, n_batch, n_features)
        # W_in_to_gates is (n_features, 4*num_units). input_dot_W is then
        # (n_time_steps, n_batch, 4*num_units).
        input_dot_W = T.dot(input, self.W_in_to_gates) + self.b_gates
        input_dot_W_img = T.dot(input, self.W_in_to_imggates) + self.b_gates

        # input_dot_w is (n_batch, n_time_steps, 4*num_units). We define a
        # slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_dot_W_n is the n'th timestep of the input, dotted with W
        # The step function calculates the following:
        #
        # i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i)
        # f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f)
        # c_t = f_tc_{t - 1} + i_t\tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)
        # o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + W_{co}c_t + b_o)
        # h_t = o_t \tanh(c_t)
        def step(input_dot_W_n, input_dot_W_img_n, cell_previous, hid_previous):

            # Calculate gates pre-activations and slice
            gates = input_dot_W_n + lasagne.nonlinearities.rectify(T.dot(hid_previous, self.W_hid_to_gates) + input_dot_W_n)
            imggates = input_dot_W_img_n + lasagne.nonlinearities.rectify(T.dot(hid_previous, self.W_hid_to_gates) + input_dot_W_img_n)
            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0) + lasagne.nonlinearities.sigmoid(slice_w(imggates, 0)) * e_imgfeats
            forgetgate = slice_w(gates, 1) + lasagne.nonlinearities.sigmoid(slice_w(imggates, 1)) * e_imgfeats
            cell_input = slice_w(gates, 2) + lasagne.nonlinearities.sigmoid(slice_w(imggates, 2)) * e_imgfeats
            outgate = slice_w(gates, 3) + lasagne.nonlinearities.sigmoid(slice_w(imggates, 3)) * e_imgfeats

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input
            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity_out(cell)
            return [cell, hid]

        def step_masked(input_dot_W_n, input_dot_W_img_n, mask, cell_previous, hid_previous):

            cell, hid = step(input_dot_W_n, input_dot_W_img_n, cell_previous, hid_previous)

            # If mask is 0, use previous state until mask = 1 is found.
            # This propagates the layer initial state when moving backwards
            # until the end of the sequence is found.
            not_mask = 1 - mask
            cell = cell*mask + cell_previous*not_mask
            hid = hid*mask + hid_previous*not_mask

            return [cell, hid]

        if self.backwards and mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input_dot_W, input_dot_W_img, mask]
            step_fun = step_masked
        else:
            sequences = [input_dot_W, input_dot_W_img]
            step_fun = step

        # Scan op iterates over first dimension of input and repeatedly
        # applies the step function
        output = theano.scan(step_fun, sequences=sequences,
                             outputs_info=[self.cell_init, self.hid_init],
                             go_backwards=self.backwards,
                             truncate_gradient=self.gradient_steps)[0][1]

        # Now, dimshuffle back to (n_batch, n_time_steps, n_features))
        output = output.dimshuffle(1, 0, 2)

        # if scan is backward reverse the output
        if self.backwards:
            output = output[:, ::-1, :]

        return output

class GradClip(theano.compile.ViewOp):

    def __init__(self, clip_lower_bound, clip_upper_bound):
        self.clip_lower_bound = clip_lower_bound
        self.clip_upper_bound = clip_upper_bound
        assert(self.clip_upper_bound >= self.clip_lower_bound)

    def grad(self, args, g_outs):
        def pgrad(g_out):
            g_out = T.clip(g_out, self.clip_lower_bound, self.clip_upper_bound)
            g_out = ifelse(T.any(T.isnan(g_out)), T.ones_like(g_out)*0.00001, g_out)
            return g_out
        return [pgrad(g_out) for g_out in g_outs]

gradient_clipper = GradClip(-10.0, 10.0)
#T.opt.register_canonicalize(theano.gof.OpRemove(gradient_clipper), name='gradient_clipper')

def adam(loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8,
         gamma=1-1e-8):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]

    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf

    """
    updates = []
    all_grads = theano.grad(gradient_clipper(loss), all_params)
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)
 
    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
 
        m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)
 
        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta) )
    updates.append((t, t + 1.))
    return updates

class Model:
    def __init__(self,
                 data,
                 U,
                 V,
                 num_output,
                 img_w=300,
                 hidden_size=100,
                 batch_size=32,
                 lr=0.001,
                 fine_tune_W=True,
                 optimizer='adam',
                 use_lstm=True,
                 is_bidirectional=False,
                 learn_init=False,
                 use_peepholes=True):
        self.data = data
        self.batch_size = batch_size
        img_h = MAX_LEN
        
        index = T.iscalar() 
        q = T.imatrix('q')
        y = T.imatrix('y')        
        img_idxs = T.ivector('img_idxs')
        q_seqlen = T.ivector('q_seqlen')
        #img_features = theano.shared(V, name='img_features', borrow=True)
        imgs = theano.shared(V, name='imgs', borrow=True)
        embeddings = theano.shared(U, name='embeddings', borrow=True)
        zero_vec_tensor = T.fvector()
        self.zero_vec = np.zeros(img_w, dtype=theano.config.floatX)
        self.set_zero = theano.function([zero_vec_tensor], updates=[(embeddings, T.set_subtensor(embeddings[0,:], zero_vec_tensor))])
        
        q_input = embeddings[q.flatten()].reshape((q.shape[0], q.shape[1], embeddings.shape[1]))
        
        l_in = lasagne.layers.InputLayer(shape=(batch_size, img_h, img_w))        
        if is_bidirectional:
            raise 'Bidirectional unsupported'
        else:
            if use_lstm:
                l_dropout = lasagne.layers.DropoutLayer(l_in, p=0.5)
                l_recurrent = LSTMLayer(l_dropout,
                                        hidden_size,
                                        W_in_to_ingate=lasagne.init.Orthogonal(),
                                        W_hid_to_ingate=lasagne.init.Orthogonal(),
                                        W_cell_to_ingate=lasagne.init.Orthogonal(),
                                        W_in_to_forgetgate=lasagne.init.Orthogonal(),
                                        W_hid_to_forgetgate=lasagne.init.Orthogonal(),
                                        W_cell_to_forgetgate=lasagne.init.Orthogonal(),
                                        W_in_to_cell=lasagne.init.Orthogonal(),
                                        W_hid_to_cell=lasagne.init.Orthogonal(),
                                        W_in_to_outgate=lasagne.init.Orthogonal(),
                                        W_hid_to_outgate=lasagne.init.Orthogonal(),
                                        W_cell_to_outgate=lasagne.init.Orthogonal(),
                                        W_in_to_imgingate=lasagne.init.Orthogonal(),
                                        W_in_to_imgforgetgate=lasagne.init.Orthogonal(),
                                        W_in_to_imgcell=lasagne.init.Orthogonal(),
                                        W_in_to_imgoutgate=lasagne.init.Orthogonal(),                                        
                                        nonlinearity_ingate=lasagne.nonlinearities.sigmoid,
                                        nonlinearity_forgetgate=lasagne.nonlinearities.sigmoid,
                                        nonlinearity_outgate=lasagne.nonlinearities.sigmoid,
                                        nonlinearity_cell=s_tanh,
                                        nonlinearity_out=s_tanh,
                                        backwards=False,
                                        learn_init=learn_init,
                                        peepholes=use_peepholes)
                l_recurrent = lasagne.layers.DropoutLayer(l_recurrent, p=0.5)                
            else:
                raise 'RNN Unsupported'
        
        l_combined_in = lasagne.layers.InputLayer(shape=(batch_size, hidden_size))
        if USE_SINGLE_ANSWER:
            l_out = lasagne.layers.DenseLayer(l_combined_in,
                                              num_units=num_output,
                                              W=lasagne.init.Orthogonal(),
                                              nonlinearity=lasagne.nonlinearities.softmax)
        else:
            l_out = lasagne.layers.DenseLayer(l_combined_in,
                                              num_units=num_output,
                                              W=lasagne.init.Uniform(0.025),
                                              nonlinearity=lasagne.nonlinearities.sigmoid)
            
        l_img_in = lasagne.layers.InputLayer(shape=(batch_size, 3, 224, 224))
                                             
        l_img_in = cuda_convnet.bc01_to_c01b(l_img_in)

        l_conv1 = cuda_convnet.Conv2DCCLayer(
                l_img_in,
                num_filters=32,
                filter_size=(8,8),
                stride=(4,4),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.Uniform(0.01),
                b=lasagne.init.Constant(0.1),
                dimshuffle=False
                )
        
        l_conv2 = cuda_convnet.Conv2DCCLayer(
                l_conv1,
                num_filters=64,
                filter_size=(4,4),
                stride=(2,2),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.Uniform(0.01),
                b=lasagne.init.Constant(0.1),
                dimshuffle=False
                )

        l_conv3 = cuda_convnet.Conv2DCCLayer(
                l_conv2,
                num_filters=64,
                filter_size=(3,3),
                stride=(1,1),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.Uniform(0.01),
                b=lasagne.init.Constant(0.1),
                dimshuffle=False
                )

        l_conv3 = cuda_convnet.c01b_to_bc01(l_conv3)        
        
        l_transform = lasagne.layers.DenseLayer(l_conv3,
                                                num_units=hidden_size,
                                                W=lasagne.init.Orthogonal(),
                                                nonlinearity=lasagne.nonlinearities.rectify)
        
        l_transform = lasagne.layers.DropoutLayer(l_transform, p=0.5)

        e_imgfeats = l_transform.get_output(imgs[img_idxs]).reshape((batch_size, hidden_size))
            
        e_question = l_recurrent.get_output(q_input, e_imgfeats=e_imgfeats, deterministic=False)[T.arange(batch_size), q_seqlen].reshape((q.shape[0], hidden_size))
        e_question_det = l_recurrent.get_output(q_input, e_imgfeats=e_imgfeats, deterministic=True)[T.arange(batch_size), q_seqlen].reshape((q.shape[0], hidden_size))   
        probas = l_out.get_output(e_question, determinstic=False)
        probas = T.clip(probas, 1e-7, 1.0-1e-7)
        probas_det = l_out.get_output(e_question_det, determinstic=True)
        probas_det = T.clip(probas_det, 1e-7, 1.0-1e-7)
                
        cost = T.nnet.binary_crossentropy(probas, y).sum(axis=1).mean()
        cost_det = T.nnet.binary_crossentropy(probas_det, y).sum(axis=1).mean()
       
        params = lasagne.layers.get_all_params(l_out) + lasagne.layers.get_all_params(l_recurrent)
        params += lasagne.layers.get_all_params(l_transform)
        if fine_tune_W:
            params += [embeddings]
            
        if 'adam' == optimizer:
            updates = adam(cost, params, learning_rate=lr)
        elif 'rmsprop' == optimizer:
            updates = lasagne.updates.rmsprop(cost, params)
        else:
            raise 'Unsupported optimizer'
            
        self.shared_data = {}        
        self.shared_data['q'] = theano.shared(np.zeros((batch_size, MAX_LEN), dtype=np.int32), borrow=True)
        self.shared_data['y'] = theano.shared(np.zeros((batch_size, num_output), dtype=np.int32), borrow=True)
        for key in ['q_seqlen', 'img_idxs']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size,), dtype=np.int32), borrow=True)

        givens = {
            q: self.shared_data['q'],
            y: self.shared_data['y'],
            q_seqlen: self.shared_data['q_seqlen'],
            img_idxs: self.shared_data['img_idxs']
        }
        self.train_model = theano.function([], cost, updates=updates, givens=givens, on_unused_input='warn')
        self.get_probas = theano.function([], probas_det, givens=givens, on_unused_input='warn')
        self.get_cost = theano.function([], cost_det, givens=givens, on_unused_input='warn')
        
    def get_batch(self, dataset, index, max_l=MAX_LEN):
        seqlen = np.zeros((self.batch_size,), dtype=np.int32)
        batch = np.zeros((self.batch_size, max_l), dtype=np.int32)
        data = dataset[index*self.batch_size:(index+1)*self.batch_size]
        for i,row in enumerate(data):
            row = row[:max_l]
            batch[i,0:len(row)] = row
            seqlen[i] = len(row)-1
        return batch, seqlen
    
    def set_shared_variables(self, dataset, index):
        q, q_seqlen = self.get_batch(dataset['q'], index)        
        num_rows = len(dataset['img_idxs'][index*self.batch_size:(index+1)*self.batch_size])
        
        img_idxs = np.zeros((self.batch_size,), dtype=np.int32)
        y = np.zeros((self.batch_size, dataset['y'].shape[1]), dtype=np.int32)
        img_idxs[:num_rows] = dataset['img_idxs'][index*self.batch_size:(index+1)*self.batch_size]
        y[:num_rows] = dataset['y'][index*self.batch_size:(index+1)*self.batch_size]
        
        self.shared_data['q'].set_value(q)
        self.shared_data['y'].set_value(y)
        self.shared_data['q_seqlen'].set_value(q_seqlen)
        self.shared_data['img_idxs'].set_value(img_idxs)

    def compute_probas(self, dataset, index):
        self.set_shared_variables(dataset, index)
        return self.get_probas()
    
    def compute_cost(self, dataset, index):
        self.set_shared_variables(dataset, index)
        return self.get_cost()    
    
    def compute_wups_score(self, input_gt, input_pred, thresh):
        if thresh == -1:
            our_element_membership = dirac_measure
        else:
            our_element_membership = lambda x, y: wup_measure(x, y, thresh)
        our_set_membership = lambda x, A: fuzzy_set_membership_measure(x, A, our_element_membership)
        score_list = [score_it(items2list(ta), items2list(pa), our_set_membership) for (ta, pa) in zip(input_gt, input_pred)]
        return float(sum(score_list)) / float(len(score_list))
    
    def compute_wups_scores(self, probas, y, thresholds=[0.9, 0.0]):
        y_pred = np.argmax(probas, axis=1)
        y_true = np.argmax(y, axis=1)
        y_pred_labels = [mlb.classes_[k] for k in y_pred]
        y_true_labels = [mlb.classes_[k] for k in y_true]
        
        return [self.compute_wups_score(y_true_labels, y_pred_labels, thresh) for thresh in thresholds]
    
    def compute_accuracy_score(self, probas, y):
        y_pred = np.argmax(probas, axis=1)
        y_true = np.argmax(y, axis=1)
        return np.sum(y_pred == y_true) / len(y_pred)
        
    def train(self, n_epochs=100, shuffle_batch=False):
        epoch = 0
        best_val_acc = 0
        best_val_cost = float('inf')
        
        n_train_batches = len(self.data['train']['y']) // self.batch_size
        n_val_batches = len(self.data['val']['y']) // self.batch_size
        n_test_batches = int(np.ceil(len(self.data['test']['y']) // self.batch_size))

        while (epoch < n_epochs):
            epoch += 1
            indices = range(n_train_batches)
            if shuffle_batch:
                indices = np.random.permutation(indices)
            bar = pyprind.ProgBar(len(indices), monitor=True)
            total_cost = 0
            start_time = time.time()
            for minibatch_index in indices:
                self.set_shared_variables(self.data['train'], minibatch_index)
                cost_epoch = self.train_model()
                total_cost += cost_epoch
                self.set_zero(self.zero_vec)
                bar.update()
            end_time = time.time()
            print "cost: ", (total_cost / len(indices)), " took: %d(s)" % (end_time - start_time)
            train_probas = np.concatenate([self.compute_probas(self.data['train'], i) for i in xrange(n_train_batches)])
            train_acc = self.compute_accuracy_score(train_probas, self.data['train']['y'])

            val_probas = np.concatenate([self.compute_probas(self.data['val'], i) for i in xrange(n_val_batches)])
            val_cost = np.mean([self.compute_cost(self.data['val'], i) for i in xrange(n_val_batches)])
            val_acc = self.compute_accuracy_score(val_probas, self.data['val']['y'])
            print 'epoch: %i' % epoch
            print 'train_acc: %f' % (train_acc)
            print 'val_acc: %f, val_cost: %f' % (val_acc, val_cost)
            if val_cost < best_val_cost or val_acc > best_val_acc:                
                best_val_cost = min(best_val_cost, val_cost)
                best_val_acc = max(best_val_acc, val_acc)
                test_probas = np.concatenate([self.compute_probas(self.data['test'], i) for i in xrange(n_test_batches)])
                test_acc = self.compute_accuracy_score(test_probas, self.data['test']['y'])
                print '************************* test_acc: %f' % (test_acc)
            else:
                pass
        #test_wups09, test_wups00 = self.compute_wups_scores(test_probas, self.data['test']['y'])
        #print 'test_wups0.9: %f, test_wups0.0: %f' % (test_wups09, test_wups00)
        
        return test_acc#, test_wups09, test_wups00

vocab = Counter()
for line in X_train_val + X_test:
    words = line.split()
    for word in words:
        vocab[word] += 1

USE_GLOVE = True
if USE_GLOVE:
    if os.path.isfile('glove_embeddings.pkl'):
        glove_embeddings = cPickle.load(open('glove_embeddings.pkl'))
    else:
        glove_embeddings = load_glove_vec(GLOVE_FILE, vocab)
        cPickle.dump(glove_embeddings, open('glove_embeddings.pkl', 'wb'))
    W, word_idx_map = get_W(glove_embeddings, k=300)
else:
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab, min_df=1)
    W, word_idx_map = get_W(rand_vecs, k=300)

for key in dataset.keys():
    data = dataset[key]['q']
    for i in xrange(len(data)):
        if type(data[i]) == list:
            continue
        data[i] = get_idx_from_sent(data[i], word_idx_map, k=300)

features = cPickle.load(open(FEATURES_FILE))
print features.shape

if os.path.isfile('IMAGES.pkl'):
    IMAGES = cPickle.load(open('IMAGES.pkl'))
else:
    IMAGES = np.array([preprocess_image(scipy.misc.imread(f)) for f in IMG_LIST], dtype=theano.config.floatX)
    cPickle.dump(IMAGES, open('IMAGES.pkl', 'wb'), protocol=-1)

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser()
parser.register('type','bool',str2bool)
parser.add_argument('--hidden_size', type=int, default=200, help='Hidden size')
parser.add_argument('--fine_tune_W', type='bool', default=True, help='Whether to fine-tune W')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
parser.add_argument('--shuffle_batch', type='bool', default=True, help='Shuffle batch')
parser.add_argument('--learn_init', type='bool', default=False, help='Learn initial hidden state')
parser.add_argument('--use_peepholes', type='bool', default=False, help='Use peephole connections')
parser.add_argument('--n_epochs', type=int, default=100, help='Num epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
args = parser.parse_args()
print "args: ", args

model = Model(dataset,
              W.astype(theano.config.floatX),
              np.array(IMAGES, dtype=theano.config.floatX),
              num_output=len(mlb.classes_),
              img_w=300,
              hidden_size=args.hidden_size,
              batch_size=args.batch_size,
              lr=args.lr,
              fine_tune_W=args.fine_tune_W,
              optimizer=args.optimizer,
              learn_init=args.learn_init,
              use_peepholes=args.use_peepholes
              )
model.train(args.n_epochs, shuffle_batch=args.shuffle_batch)
