# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import caffe
import numpy as np
caffe.set_mode_cpu()
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
image = caffe.io.load_image('/home/zxd/projects/darknet-master/data/dog.jpg')
# print image.shape
# print image.dtype
transformer = caffe.io.Transformer({'data': (1, 3, 416, 416)})

transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
# transformer.set_mean('data', )            # subtract the dataset-mean value in each channel
# transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
# transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
transformed_image = transformer.preprocess('data', image)
# np.save('/home/zxd/data/layer_output_caffe/image', image)
# array_from_file = np.load('/home/zxd/data/layer_output_caffe/image.npy');
# print array_from_file.shape
# create transformer for the input called 'data'
np.save('/home/zxd/data/layer_output_caffe/sized_image', transformed_image)
print transformed_image.shape
