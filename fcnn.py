# Author: Haonan Tian 
# Date: 07/27/2018
# All Rights Reserved 

################################# Description ########################################
# This is a demo of fully connected neural network. The data set of this model can be 
# found at the following site:
# http://yann.lecun.com/exdb/mnist/
######################################################################################

################################# Initialization ########################################
import time
import csv 
import math
import struct as st 
import numpy as np 

######################################################################################

################################# Load IDX Data ########################################
# This load data function only designed to load binary data from mniset data set 
def loadData(input_data, input_label): 
	filename = {'images' : input_data ,'labels' : input_label}
	# Load image file 
	imagesfile = open(filename['images'],'rb')

	imagesfile.seek(0) # set offset at the beginning 
	magic_image = st.unpack('>4B',imagesfile.read(4)) # return the magic number  

	nImg = st.unpack('>I',imagesfile.read(4))[0] #num of images
	nR = st.unpack('>I',imagesfile.read(4))[0] #num of rows
	nC = st.unpack('>I',imagesfile.read(4))[0] #num of column

	images_array = np.zeros((nImg,nR,nC))

	nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
	images_array = np.asarray(st.unpack('>'+'B'*nBytesTotal,imagesfile.read(nBytesTotal))).reshape((nR*nC, nImg))

	# Load label file
	labelfile = open(filename['labels'], 'rb')

	labelfile.seek(0)
	magic_label = st.unpack('>4B', labelfile.read(4))

	num_Item = st.unpack('>I',labelfile.read(4))[0]

	labels_array = np.zeros((num_Item, 1))

	labelTotalBytes = num_Item * 1

	labels_array = np.asarray(st.unpack('>' + 'B' * labelTotalBytes, labelfile.read(labelTotalBytes))).reshape((num_Item,1))

	#print(labels_array.shape)
	#print(images_array.shape)
	return images_array, labels_array, nR * nC # nR*nC equals to the number of input dimensions

def loadData_test(input_data, input_label): 
	filename = {'images' : input_data ,'labels' : input_label}
	# Load image file 
	imagesfile = open(filename['images'],'rb')

	imagesfile.seek(0) # set offset at the beginning 
	magic_image = st.unpack('>4B',imagesfile.read(4)) # return the magic number  

	nImg = st.unpack('>I',imagesfile.read(4))[0] #num of images
	nImg = 2
	nR = st.unpack('>I',imagesfile.read(4))[0] #num of rows
	nC = st.unpack('>I',imagesfile.read(4))[0] #num of column

	images_array = np.zeros((nImg,nR,nC))

	nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
	images_array = np.asarray(st.unpack('>'+'B'*nBytesTotal,imagesfile.read(nBytesTotal))).reshape((nR*nC, nImg))

	# Load label file
	labelfile = open(filename['labels'], 'rb')

	labelfile.seek(0)
	magic_label = st.unpack('>4B', labelfile.read(4))

	num_Item = st.unpack('>I',labelfile.read(4))[0]
	num_Item = 2

	labels_array = np.zeros((num_Item, 1))

	labelTotalBytes = num_Item * 1

	labels_array = np.asarray(st.unpack('>' + 'B' * labelTotalBytes, labelfile.read(labelTotalBytes))).reshape((num_Item,1))

	#print(labels_array.shape)
	#print(images_array.shape)
	return images_array, labels_array, nR * nC # nR*nC equals to the number of input dimensions

def load_csv_data(input_data):
	file_data = open(input_data)
	data_reader = csv.reader(file_data)
	data = [r for r in data_reader]
	label_array = np.zeros((len(data), 1), dtype = int)
	for i in range(len(data)):
		value = int(data[i].pop(0))
		label_array[i,0] = value
	file_data.close()
	#print(label_array)
	data_array = np.asarray(data, dtype = float).reshape((len(data[0]), len(data)))
	return data_array, label_array, len(data[0])

################################# Fully Connected Module ########################################
def initial_parameters(dims): # input dimension contains the dimension of input layer
	num_layer = len(dims)
	parameters = {}
	for i in range(num_layer-1):
		parameters['W' + str(i+1)] = np.random.randn(dims[i+1], dims[i]) * 0.01
		parameters['b' + str(i+1)] = np.zeros((dims[i+1], 1))
		#print("W = " + str(parameters['W' + str(i+1)].shape))
	return parameters

def initial_grads(dims):
	num_layer = len(dims)
	grad = {}
	for i in range(num_layer-1):
		grad['dW' + str(i+1)] = np.zeros((dims[i+1], dims[i]))
		grad['db' + str(i+1)] = np.zeros((dims[i+1],1))
		#print("dW = " + str(grad['dW' + str(i+1)].shape))
	return grad

def activation_tanh(Z):
	A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
	return A 

def activation_sigmoid(Z):
	A = 1 / (1 + np.exp(-Z))
	return A

def activation_relu(Z):
	A = Z
	A[np.where(A<0)] = 0
	return A

def softmax(Z):
	A = np.exp(Z) / np.sum(np.exp(Z), axis = 0)
	return A

def move_forward(X, parameters):
	num_layer = math.floor(len(parameters)/2)
	Z1 = np.dot(parameters['W1'], X) + parameters['b1']
	A1 = np.tanh(Z1)
	Z2 = np.dot(parameters['W2'], A1) + parameters['b2']
	A2 = np.tanh(Z2)
	Z3 = np.dot(parameters['W3'], A2) + parameters['b3']
	A3 = np.tanh(Z3)
	Z4 = np.dot(parameters['W4'], A3) + parameters['b4']
	A4 = activation_sigmoid(Z4)
	caches = {'A1': A1, 'A2': A2, 'A3': A3, 'A4': A4}
	return A4, caches

def convert_y(y, different_labels):
	m = y.shape[0]
	output = np.zeros((different_labels, m))
	for i in range(m):
		value = y[i,0]
		output[value,i] = 1
	return output

def compute_cost(A4, y):
	m = A4.shape[1]
	temp = np.multiply(y, np.log(A4)) + np.multiply((1 - y), np.log(1 - A4))
	return -1/m * np.sum(temp)

def tanh_derivative(A):
	dA = (1 - np.power(A, 2))
	return dA

def move_back(parameters, X, y, caches):
	m = X.shape[1]
	# Unpack cached value
	A1 = caches['A1']
	A2 = caches['A2']
	A3 = caches['A3']
	A4 = caches['A4']

	W1 = parameters['W1']
	W2 = parameters['W2']
	W3 = parameters['W3']
	W4 = parameters['W4']
	# Back propagation 
	dZ4 = 2 * (A4 - y) * A4 * (1-A4)
	dW4 = (1/m) * np.dot(dZ4, A3.T)
	db4 = (1/m) * np.sum(dZ4, axis = 1, keepdims = True)
	#print("dW4 = " + str(dW4.shape))
	#print('db4 = ' + str(db4.shape))

	dZ3 = np.dot(W4.T, dZ4) * (1 - np.power(A3, 2))
	dW3 = (1/m) * np.dot(dZ3, A2.T)
	db3 = (1/m) * np.sum(dZ3, axis = 1, keepdims = True)
	#print("dW3 = " + str(dW3.shape))
	#print('db3 = ' + str(db3.shape))

	dZ2 = np.dot(W3.T, dZ3) * (1 - np.power(A2, 2))
	dW2 = (1/m) * np.dot(dZ2, A1.T)
	db2 = (1/m) * np.sum(dZ2, axis = 1, keepdims = True)
	#print("dW2 = " + str(dW2.shape))
	#print('db2 = ' + str(db2.shape))

	dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
	dW1 = (1/m) * np.dot(dZ1, X.T)
	db1 = (1/m) * np.sum(dZ1, axis = 1, keepdims = True)
	#print("dW1 = " + str(dW1.shape))
	#print('db1 = ' + str(db1.shape))

	# pack the data
	grads = {'dW1': dW1, 'dW2': dW2, 'dW3': dW3, 'dW4': dW4,
			 'db1': db1, 'db2': db2, 'db3': db3, 'db4': db4}
	return grads

def update_parameter(grad, parameters, learning_rate):
	# Unpack the data
	W1 = parameters['W1']
	W2 = parameters['W2']
	W3 = parameters['W3']
	W4 = parameters['W4']
	b1 = parameters['b1']
	b2 = parameters['b2']
	b3 = parameters['b3']
	b4 = parameters['b4']

	# unpack gradients
	dW1 = grad['dW1']
	db1 = grad['db1']
	dW2 = grad['dW2']
	db2 = grad['db2']
	dW3 = grad['dW3']
	db3 = grad['db3']
	dW4 = grad['dW4']
	db4 = grad['db4']

	# update parameters
	#print("W1 " + str(W1.shape) + " dW1 " + str(dW1.shape))
	W1 = W1 - learning_rate * dW1
	b1 = b1 - learning_rate * db1
	W2 = W2 - learning_rate * dW2
	b2 = b2 - learning_rate * db2
	W3 = W3 - learning_rate * dW3
	b3 = b3 - learning_rate * db3
	W4 = W4 - learning_rate * dW4
	b4 = b4 - learning_rate * db4

	# pack the data
	parameters = {'W1': W1, 'W2': W2, 'W3': W3, 'W4': W4,
				  'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4}
	return parameters

def predict(X, parameters, y):
	A4, _ = move_forward(X, parameters)
	print(A4)
	max_array = np.argmax(A4, axis = 0)
	match = np.zeros((y.shape[0],y.shape[1]))
	#print("match shape = " + str(match.shape))
	#print("max_array shape = " + str(max_array.shape))
	#print(max_array)
	for i in range(y.shape[1]):
		match[max_array[i], i] = 1

	result = np.count_nonzero(np.multiply(match, y))
	total = y.shape[1]
	return result / total

def fcnn_model(X, y_converted, dims, different_labels, num_iterations, learning_rate):
	# initializa necessary parameters and variables 
	parameters = initial_parameters(dims)
	grads = initial_grads(dims)

	# start iterations
	for i in range(num_iterations):
		# perform forward propagation
		A4, caches = move_forward(X, parameters)

		# compute cost
		#y_converted = convert_y(y, different_labels) # convert y for cost calculation
		cost = compute_cost(A4, y_converted)

		print("cost for iterations " + str(i) + " is " + str(cost) + "\n")

		# perform back propagation
		grads = move_back(parameters, X, y_converted, caches)

		# update parameters with the gradients
		parameters = update_parameter(grads, parameters, learning_rate)

	return parameters

def convert_to_print(array):
	result = ''
	for j in range(5):
		for i in range(784):
			result = result + str(array[i,j]) + ','
		print(result.strip(',') + "\n! " + str(j) + " !\n")
		result = ''

######################################################################################
def main():
	data_array, label_array, m = loadData(train_data, train_label)
	data_array, label_array, m = load_csv_data('mnist_train.csv')	
	print("Finish loading the data\n")
	print("m = " + str(m))
	updated_dims = [m] + dims
	y_converted = convert_y(label_array, different_labels)
	parameters = fcnn_model(data_array, y_converted, updated_dims, different_labels, num_iterations, learning_rate)
	#y_converted = convert_y(y, different_labels)
	#data_test, label_test, m_test = loadData_test(test_data, test_label)
	data_test, label_test, m_test = load_csv_data('mnist_test.csv')
	y = convert_y(label_test, different_labels)
	result = predict(data_test, parameters, y)
	print("Finished running the training data with accuracy = " + str(result) + "\n")
	return 0

if __name__ == "__main__":
	train_data = "mniset_train_data"
	train_label = "mniset_train_label"
	test_data = "mniset_test_data"
	test_label = "mniset_test_label"
	different_labels = 10
	dims = [200, 40, 20, 10]
	num_iterations = 40
	learning_rate = 1
	main()