#importing numpy for making the operation
import numpy as np
#import library for turning output to a png
import imageio
# Load files
input = np.load('samples_2.npy')
kernel = np.load('kernel.npy')
# Define my_conv2d 
def my_conv2d(input, kernel):
batch_size, input_channels, input_height, input_width = input.shape
output_channels, input_channels, filter_height, filter_width = 
kernel.shape
# Initialize output tensor with zeros
output_height = input_height - filter_height + 1
output_width = input_width - filter_width + 1
output = np.zeros((batch_size, output_channels, output_height, 
output_width))
# Perform convolution operation
for b in range(batch_size):
for c_out in range(output_channels):
for h_out in range(output_height):
for w_out in range(output_width):

# Extract input patch
input_patch = input[b, :, h_out:h_out+filter_height, 
w_out:w_out+filter_width]
# Compute dot product between input patch and kernel
output[b, c_out, h_out, w_out] = np.sum(input_patch * 
kernel[c_out]) 
return output
# Pass my_conv2d function with input and kernel to out
out = my_conv2d(input, kernel)
# Save the output to a file
np.save('out.npy', out)
# Convert the output to a grayscale image 
# Save in .PNG format
output_image = np.mean(out, axis=1)[0].squeeze()
imageio.imwrite('output.png', output_image)
# Yasincan Bozkurt