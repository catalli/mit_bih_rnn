import numpy as np
import pickle
import matplotlib.pyplot as plt
import keras 
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors as NNeigh
from keras.models import Sequential, Model
from keras.layers import Dense, Convolution2D, Activation, Flatten, MaxPooling2D,Input,Dropout,GlobalAveragePooling2D
from keras import backend as K
from keras.datasets import cifar10, cifar100,mnist
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.engine.topology import Layer
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import initializers
import tensorflow as tf
import time
np.random.seed(1234)
def print_activations(t):
  print (t.op.name, t.get_shape().as_list())

def string_match(s1,s2):
	l1=len(s1)
	l2=len(s2)
	if(l1>l2):
		return False
	s2=s2[0:l1]
	return s1==s2
class cluster(object):
	#this object clusters a set of weights in a hierarchical tree
	#num_clusters: No.clusters in each level of the tree. Should be 2^x
	#encoding offset: used for recursive calls. Pass 0 or don't pass anything! it works automatically
	#depth: used to determine internal parameters with recursive calls. Pass zero or don't pass anything.
	#original_weights: the weights to be clustered
	def __init__(self,num_clusters,original_weights,depth=0,encoding_offset=0):
		self.depth=depth
		num_clusters=np.ravel(num_clusters)
		self.num_clusters=num_clusters[0]
		self.original_weights=original_weights
		self.original_shape=original_weights.shape
		#print self.original_weights.shape
		w=self.original_weights
		est=KMeans(n_clusters=self.num_clusters)
		to_cluster=np.reshape(w.astype(np.float64),(-1,1))
		if to_cluster.shape[0]>=self.num_clusters:
			#this is a valid clustering where #clusters<= #datapoints
			est.fit(to_cluster)
			self.centroids=np.ravel(est.cluster_centers_)
			self.near=NNeigh(n_neighbors=1)
			self.near.fit(est.cluster_centers_)
			#print est.cluster_centers_.shape
		else:
			#in this case the number of datapoints is less than clusters, simply add zeros in such cases!
			self.centroids=np.ravel(to_cluster)
			num_zeros=self.num_clusters-self.centroids.shape[0]
			zeros=np.zeros(num_zeros)
			self.centroids=np.concatenate((self.centroids,zeros))
			self.near=NNeigh(n_neighbors=1)
			self.near.fit(np.reshape(self.centroids,(-1,1)))

		self.encoding_offset=encoding_offset
		w=np.reshape(w,(-1,1))
		#print self.encoding_offset
		self.encodings=self.near.kneighbors(w)[1]
		#print self.encodings.shape
		self.encodings=self.encodings.reshape(self.original_shape)+self.encoding_offset
		#print np.unique(self.encodings),self.centroids
		if num_clusters.shape[0]==1:
			self.num_clusters_next_levels=None
		else:
			self.num_clusters_next_levels=num_clusters[1:]
			self.add_sub_clusters()
		

		
			
	def add_sub_clusters(self):
		#this function clusters each sub_group into "num_clusters" clusters
		#first separate the original weights to the encoded clusters:
		self.sub_clusters=[]
		for i in range(self.num_clusters):
			
			this_sub_cluster_weights=self.original_weights[self.encodings==(i+self.encoding_offset)]
			self.sub_clusters.append(cluster(num_clusters=self.num_clusters_next_levels,
				encoding_offset=(self.encoding_offset+i)*self.num_clusters_next_levels[0],
				original_weights=this_sub_cluster_weights,
				depth=self.depth+1))
	
	def tree_search_nn(self,w,depth=0):
		#this function retrieves the colosest clusters from a specific depth
		#w: the weights to be clustered. weights should flattened
		#depth: the depth from which the cluster centroids are picked.
		#returns: the cluster centroids and encoded weights
		#original_shape=w.shape
		w=np.reshape(w,(-1,1))
		encodings=self.near.kneighbors(w)[1]
		if depth==self.depth:
			return encodings.ravel()+self.encoding_offset,self.centroids
		else:
			encoding_up=np.zeros(w.shape[0])
			centroids_up=np.asarray([])
			for i in range(self.num_clusters):
				w_part=w[encodings==i]
				#print w_part.shape
				encoding_down,centroids_down=self.sub_clusters[i].tree_search_nn(w_part,depth)
				encoding_up[np.ravel(encodings)==i]=encoding_down
				centroids_up=np.concatenate((centroids_up,centroids_down),axis=0)
			return encoding_up.ravel().astype(np.int32),centroids_up

class quantize_dense_ (Layer):
	def __init__(self, codebook, exact_quantized,**kwargs):
		self.exact_quantized=tf.Variable(tf.cast(exact_quantized, tf.bool),trainable=False)
		self.codebook=tf.Variable(tf.to_float(codebook),trainable=False);
		super(quantize_dense_, self).__init__(**kwargs)
	def set_codebook(self,codebook):
		sess=K.get_session()
		sess.run(self.codebook.assign(codebook))
	def set_exact_quantized(self,exact_quantized):
		sess=K.get_session()
		sess.run(self.exact_quantized.assign(exact_quantized))
	def call(self, x, mask=None):
		codebook_length=int(self.codebook.get_shape()[0])
		vector_len=int(x.get_shape()[1])
		x_reshaped=tf.reshape(x,[-1,1,int(x.get_shape()[1])])
		#print_activations(x)

		tiled=tf.tile(x_reshaped,[1,codebook_length,1])
		#print_activations(tiled)
		#print_activations(codebook)
		codebook_reshaped=tf.reshape(self.codebook,[codebook_length,1])
		codebook_tiled=tf.tile(codebook_reshaped,[1,vector_len])
		#print_activations(codebook_tiled)
		abs_=tf.abs(tiled-codebook_tiled)
		#print_activations(abs_)
		agmin=tf.argmin(abs_,axis=1)
		#print_activations(agmin)
		quantized=tf.gather(self.codebook,agmin)
		
		inp_cond=tf.cond(self.exact_quantized,lambda:x,lambda:quantized)
		#print inp_cond.get_shape()
		return inp_cond
	def get_output_shape_for(self,input_shape):
		return input_shape	

class quantize_conv_ (Layer):
	def __init__(self, codebook,exact_quantized,**kwargs):
		self.exact_quantized=tf.Variable(tf.cast(exact_quantized, tf.bool),trainable=False)
		self.codebook=tf.Variable(tf.to_float(codebook),trainable=False);
		super(quantize_conv_, self).__init__(**kwargs)
	def set_codebook(self,codebook):
		sess=K.get_session()
		sess.run(self.codebook.assign(codebook))
	def set_exact_quantized(self,exact_quantized):
		sess=K.get_session()
		sess.run(self.exact_quantized.assign(exact_quantized))
	def call(self, x, mask=None):
		width=int(x.get_shape()[1])
		height=int(x.get_shape()[2])
		num_channels=int(x.get_shape()[3])
		codebook_len=int(self.codebook.get_shape()[0])

		reshaped=tf.reshape(tensor=x,shape=[-1,width,height,1,num_channels])
		tiled=tf.tile(reshaped,[1,1,1,codebook_len,1])
		#print_activations(tiled)

		codebook_reshaped=tf.reshape(self.codebook,shape=[1,1,codebook_len,1])
		codebook_tiled=tf.tile(codebook_reshaped,[width,height,1,num_channels])
		#print_activations(codebook_tiled)
		abs_=tf.abs(tiled-codebook_tiled)
		#print_activations(abs_)
		agmin=tf.argmin(abs_,axis=3)
		#print_activations(agmin)
		quantized=tf.gather(self.codebook,agmin)
		#print_activations(quantized)
		inp_cond=tf.cond(self.exact_quantized,lambda:x,lambda:quantized)
		return inp_cond
	def get_output_shape_for(self,input_shape):
		return input_shape

def quantize_weights(layer,clusters_per_level,depth,alltogether=True):
	#quantizes weights of the layer based on clusters_per_level
	#if altogether is set to true (dense layers), all the rows of the matrix have same codebooks
	#otherwise (conv layers), each output channel has a different codebook for its corresponding 3-D kernel
	#print 'bucketizing parameters of '+layer.name
	w,b=layer.get_weights()
	if alltogether:
		to_cluster=np.ravel(w)
		if to_cluster.shape[0]>100000:
			#randomly sample 100000:
			to_cluster=np.random.permutation(to_cluster)[0:100000]
		cl=cluster(clusters_per_level,to_cluster)
		enc,cent=cl.tree_search_nn(w,depth)
		w_clustered=cent[enc].reshape(w.shape)
		layer.set_weights([w_clustered,b])
	else:
		for i in range(w.shape[-1]):
			cl=cluster(clusters_per_level,w[:,:,:,i])
			enc,cent=cl.tree_search_nn(w[:,:,:,i],depth)
			w_clustered=cent[enc].reshape(w.shape[0:-1])
			w[:,:,:,i]=w_clustered
		layer.set_weights([w,b])
def layer_codebook_finder(x,input,output,clusters_per_level,depth):
	#print 'finding activation codebook for '+layer.name
	m=Model(input=input,output=output)
	real_vals=m.predict(x)
	to_cluster=np.ravel(real_vals)
	if to_cluster.shape[0]>100000:
		#randomly sample 100000:
		to_cluster=np.random.permutation(to_cluster)[0:100000]
	cl=cluster(clusters_per_level,to_cluster)
	enc,cent=cl.tree_search_nn(real_vals,depth)
	return cent
def bucketize_model_params(model,clusters_per_level,depth):
	#clusters per layer: number of clusters in each level of the tree
	#depth: the depth from which the leaves are picked
	for layer in model.layers:
		if string_match('convolution2d',layer.name):		
			quantize_weights(layer,clusters_per_level=clusters_per_level,depth=depth,alltogether=False)
			#codebook=layer_codebook_finder(X_train[0:100],input=model.input,output=model.output,clusters_per_level=conv_clusters_act,depth=conv_depth)
			#model.add(quantize_conv_(codebook,False))
		if string_match('dense',layer.name):
			quantize_weights(layer,clusters_per_level=clusters_per_level,depth=depth,alltogether=True)
			#codebook=layer_codebook_finder(X_train[0:100],input=model.input,output=model.output,clusters_per_level=dense_clusters_act,depth=dense_depth)
			#model.add(quantize_dense_(codebook,False))


def model_exact_or_quantize(model,exact_quantized):
	#disable or enable activation quantization for the whole model
	for layer in model.layers:
		if string_match('quantize_dense_',layer.name) or string_match('quantize_conv_',layer.name):
			layer.set_exact_quantized(exact_quantized)


'''

batch_size = 32
num_classes = 19
epochs = 100


dataset='DAS_o'
Train=False
# the data, shuffled and split between train and test sets
if dataset=='mnist':
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
if dataset=='Isolet_o' or dataset=='DAS_o' or dataset=='InnLoc_NB':
	foo=open('../data/'+dataset+'.pkl','rb')
	data=pickle.load(foo)
	X_train=np.concatenate((data[0][0],data[1][0]))
	y_train=np.concatenate((data[0][1],data[1][1])).astype(np.int32)
	X_test=data[2][0]
	y_test=data[2][1].astype(np.int32)
	minim=np.min(y_train)
	y_train=y_train-minim
	y_test=y_test-minim

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
if dataset=='mnist':
	X_train = X_train.reshape(60000, 784)/255
	X_test = X_test.reshape(10000, 784)/255
elif dataset=='DAS_o':
	var=np.var(X_train,axis=0)
	ind=np.where(var==0)
	var[ind]=1;
	mean=np.mean(X_train,axis=0)
	for i in range(X_train.shape[0]):
		X_train[i]=(X_train[i]-mean)/var
	for i in range(X_test.shape[0]):
		X_test[i]=(X_test[i]-mean)/var


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')




if Train:
	model=Sequential()
	model.add(Dense(512,activation='relu',input_shape=X_train.shape[1:]))
	model.add(Dropout(0.5))
	model.add(Dense(512,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes,activation='softmax'))
				
	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',
		          optimizer=sgd,
		          metrics=['accuracy'])
	
	model.fit(X_train, Y_train,
				batch_size=batch_size,
				nb_epoch=epochs,
				validation_data=(X_test, Y_test),
				verbose=2)
	model.save(dataset+'.h5')
else:
	accs=[]
	model1=load_model(dataset+'.h5')
	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	model1.compile(loss='categorical_crossentropy',
		          optimizer=sgd,
		          metrics=['accuracy'])
	
	orig_acc=model1.evaluate(x=X_test,y=Y_test)[1]
	print 'baseline accuracy is:',orig_acc
	#model1.summary()
	for clusters_per_level in [[2],[4],[2,4],[2,8],[4,8],[4,16]]:
		depth=len(clusters_per_level)-1
		
		model=Sequential()
		cl=cluster(clusters_per_level,X_train[0:500])
		IdontCare,codebook=cl.tree_search_nn(X_train[0:1],depth)
		model.add(quantize_dense_(codebook,True,input_shape=X_train.shape[1:],trainable=False))
		for i in range(len(model1.layers)): 
			layer=model1.layers[i]
			#print layer.name
			if string_match('convolution2d',layer.name):		
				model.add(Convolution2D(nb_filter=layer.nb_filter, 
					nb_row=layer.nb_row, 
					nb_col=layer.nb_col, 
					border_mode=layer.border_mode,
					activation=layer.activation,
					weights=layer.get_weights(),
					input_shape=X_train.shape[1:]))
				#quantize_weights(model.layers[-1],clusters_per_level=clusters_per_level,depth=depth)
				codebook=layer_codebook_finder(X_train[0:100],input=model.input,output=model.output,clusters_per_level=clusters_per_level,depth=depth)
				model.add(quantize_conv_(codebook,False,trainable=False))
			if string_match('maxpooling2d',layer.name):
				model.add(MaxPooling2D(pool_size=layer.pool_size,strides=layer.strides))
				
			if string_match('dense',layer.name):
				model.add(Dense(output_dim=layer.output_dim,
								weights=layer.get_weights(),
								activation=layer.activation))
				#quantize_weights(model.layers[-1],clusters_per_level=clusters_per_level,depth=depth)
				codebook=layer_codebook_finder(X_train[0:100],input=model.input,output=model.output,clusters_per_level=clusters_per_level,depth=depth)
				model.add(quantize_dense_(codebook,False,trainable=False))
				
			if string_match('flatten',layer.name):
				model.add(Flatten())
				
			if string_match('dropout',layer.name):
				model.add(Dropout(0.5))
				
		#model.summary()
		sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='categorical_crossentropy',
			          optimizer=sgd,
			          metrics=['accuracy'])
		
		accs_this=[]
		dict_={}
		for clusters_per_level_weight in [[2],[4],[2,4],[2,8],[4,8],[8,8]]:
			
			depth=len(clusters_per_level_weight)-1
			for i in range(1):
				t0_iter=time.time()
				#print i
				#disable activation quantization:
				#
				#print 'weights original, activations original:',model.evaluate(x=X_test,y=Y_test)
				"""if i==0:
														w=model.layers[-2].get_weights()[0]
														w=np.ravel(w)
														dict_['before_cluster']=w
														h,b=np.histogram(w,bins=1000)
														plt.bar(b[0:-1],h,width=0.001)
														plt.show()
										"""
				model_exact_or_quantize(model,False)
				bucketize_model_params(model,clusters_per_level_weight,depth)
				"""if i==0:
														w=model.layers[-2].get_weights()[0]
														w=np.ravel(w)
														dict_['after_cluster']=w
														h,b=np.histogram(w,bins=1000)
														plt.bar(b[0:-1],h,width=0.001)
														plt.show()"""
				#print 'weights quantized, activations original:',model.evaluate(x=X_test,y=Y_test)
				#model_exact_or_quantize(model,False)
				acc=model.evaluate(x=X_test,y=Y_test,verbose=0)
				#errs.append(1-acc[1])
				#print 'weights quantized, activations quantized:',acc
				#model_exact_or_quantize(model,True)
				# Fit the model on the batches generated by datagen.flow().
				model_exact_or_quantize(model,True)
				model.fit(X_train, Y_train,
									batch_size=batch_size,
									nb_epoch=1,
									validation_data=(X_test, Y_test),
									verbose=0)
				"""if i==0:
														w=model.layers[-2].get_weights()[0]
														w=np.ravel(w)
														dict_['after_retrain']=w
														h,b=np.histogram(w,bins=1000)
														plt.bar(b[0:-1],h,width=0.001)
														plt.show()"""
				t1_iter=time.time()
			#dict_['errs']=errs
			accs_this.append(orig_acc-acc[1])
			print 'activation:',clusters_per_level,'weight:',clusters_per_level_weight,'err:',orig_acc-acc[1],'delay:',5*(t1_iter-t0_iter)
		accs.append(accs_this)
	foo=open(dataset+'.pkl','wb')
	#pickle.dump(accs,foo)
	foo.close()
'''
