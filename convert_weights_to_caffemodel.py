# -*- coding: utf-8 -*-
import caffe
import numpy as np
caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()
model_filename = 'yolo.prototxt'
yoloweight_filename = 'yolo.weights'
caffemodel_filename = 'yolo.caffemodel'
print 'model file is ', model_filename
print 'weight file is ', yoloweight_filename
print 'output caffemodel file is ', caffemodel_filename
net = caffe.Net(model_filename, caffe.TEST)
net.forward()
# for each layer, show the output shape
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)
count = 0
for layer_name, param in net.params.iteritems():
    sum = param[0].data.size + param[1].data.size
    count += sum
    print layer_name + '\t' + str(param[0].data.shape) +\
          '\t' + str(param[1].data.shape) + '\t' + str(sum)
print 'count=', str(count)
params = net.params.keys()
# read weights from file and assign to the network
netWeightsInt = np.fromfile(yoloweight_filename, dtype=np.int32)
transFlag = (netWeightsInt[0]>1000 or netWeightsInt[1]>1000)
# transpose flag, the first 4 entries are major, minor, revision and net.seen
print 'transFlag = %r' % transFlag
netWeightsFloat = np.fromfile(yoloweight_filename, dtype=np.float32)
netWeights = netWeightsFloat[4:]
# start from the 5th entry, the first 4 entries are major, minor, revision and net.seen
print netWeights.shape
count = 0
for pr in params:
    lidx = list(net._layer_names).index(pr)
    layer = net.layers[lidx]
    if count == netWeights.shape[0]:
        print "WARNING: no weights left for %s" % pr
        break
    if layer.type == 'Convolution':
        print pr+"(conv)"
        # bias
        if len(net.params[pr]) > 1:
            bias_dim = net.params[pr][1].data.shape
        else:
            bias_dim = (net.params[pr][0].data.shape[0], )
        biasSize = np.prod(bias_dim)
        conv_bias = np.reshape(netWeights[count:count+biasSize], bias_dim)
        if len(net.params[pr]) > 1:
            assert(bias_dim == net.params[pr][1].data.shape)
            net.params[pr][1].data[...] = conv_bias
            conv_bias = None
        count += biasSize
        # batch_norm
        next_layer = net.layers[lidx+1]
        # weights
        dims = net.params[pr][0].data.shape
        weightSize = np.prod(dims)
        net.params[pr][0].data[...] = np.reshape(netWeights[count:count+weightSize], dims)
        count += weightSize
    else:
        print "WARNING: unsupported layer, "+pr
if np.prod(netWeights.shape) != count:
    print "ERROR: size mismatch: %d" % count
# net.save(caffemodel_filename)
