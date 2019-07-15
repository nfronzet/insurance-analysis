import matplotlib.pyplot as plt
import imageio
import numpy as np

class ConvolutionalOperation:
    def apply3x3kernel(self, image, kernel): # Simple 3x3 kernel operation
        newimage=np.array(image)
        for m in range(1,image.shape[0]-2):
            for n in range(1,image.shape[1]-2):
                newelement = 0
                for i in range(0, 3):
                    for j in range(0, 3):
                        newelement = newelement + image[m - 1 + i][n - 1+ j]*kernel[i][j]
                newimage[m][n] = newelement
        return (newimage)
  
class PoolingOperation:
    def apply2x2pooling(self, image, stride): # Simple 2x2 kernel operation
        newimage=np.zeros((int(image.shape[0]/2),int(image.shape[1]/2)),np.float32)
        for m in range(1,image.shape[0]-2,2):
            for n in range(1,image.shape[1]-2,2):
                newimage[int(m/2),int(n/2)] = np.max(image[m:m+2,n:n+2])
        return (newimage)

def kernel_samples(filepath):
    kernels = {"Identity":[[0, 0, 0], [0., 1., 0.], [0., 0., 0.]],
        "Laplacian":[[0, -1, 0.], [-1, 4, -1], [0, -1, 0]],
        "Left Sobel":[[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]],
        "Upper Sobel":[[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]}
    
    arr = imageio.imread(filepath) [:,:,0].astype(np.float)
    print('Image loaded')
    
    conv = ConvolutionalOperation()
    plt.figure(figsize=(30,30))
    fig, axs = plt.subplots(figsize=(30,30))
    j = 1
    print('Starting iterative kernel applications')
    for key,value in kernels.items(): #change this to switch between kernel dicts
        print('Starting application of kernel ' + str(j) + ' out of ' + str (len(kernels)))
        axs = fig.add_subplot(2,2,j) #change this when plotting multiple kernel results
        out = conv.apply3x3kernel(arr, value)
        plt.imshow(out, cmap=plt.get_cmap('binary_r'))
        print('Application of kernel ' + str(j) + ' out of ' + str (len(kernels)) + ' complete')
        j = j + 1
    plt.show()
