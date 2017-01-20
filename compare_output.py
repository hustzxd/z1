import numpy as np
import sys

import struct

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: python %s <output_in_caffe> <output_in_darknet>" % sys.argv[0]
        exit(1)
    caffe_filename = sys.argv[1]
    darknet_filename = sys.argv[2]
    output_caffe = np.load(caffe_filename)
    print 'ouput_caffe shape: ' + str(output_caffe.shape)
    output_caffe = output_caffe.reshape(-1, )
    fileData = open(darknet_filename, 'rb')
    print 'comparing...'
    sum = 0
    error = 0
    for one_in_caffe in output_caffe:
        sum += 1
        one_in_darknet = struct.unpack('f', fileData.read(4))
        if abs(one_in_caffe - one_in_darknet) / max(one_in_caffe, one_in_caffe) > 0.08:
            error += 1
    rate = 1 - float(error) / sum
    print '%d / %d = %f' % (sum - error, sum, rate)
