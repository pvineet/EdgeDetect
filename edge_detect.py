
import os
import numpy as np
import inspect
from utils import gauss2D, LoGKernel, DoGKernel, non_max_suppression, hysteresis
from PIL import Image
from scipy.ndimage.filters import convolve
from scipy.misc import imsave

class EdgeDetector():
    def __init__(self, show_output = True, apply_gaussian = True):
        self.in_image = None
        self.display_output = show_output
        self.output_dir = "out"
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        self.apply_gaussian = apply_gaussian
        self.gaussian_kernel = np.array(gauss2D(shape=(5,5), sigma=1.0))
        self.min_pixel_val = 0
        self.max_pixel_val = 255
        self.BHT_enable = True
        if self.BHT_enable:
            self.dilate_enable = True
            self.erode_enable = True
            self.open_edges_enable = True
            self.close_edges_enable = True
        else:
            self.dilate_enable = False
            self.erode_enable = False
            self.open_edges_enable = False
            self.close_edges_enable = False

    def conv(self, arr, imfilter):
        """
            Convolution function
            @params: Input image
            @params: Filter
            @return: Convolved image
        """
        return convolve(arr, imfilter)

    def display(self, img):
        """
            Display Image
            @params: Input RGB image.
        """
        display = Image.fromarray(img)
        display.show()

    def save_image(self, img, fname):
        img = Image.fromarray(img)
        fname = fname + '.jpg'
        print("Saving", fname)
        imsave(fname, img)

    def find_threshold(self, img):
        """
            Find Threshold Value
            @params: Input edge intensity array.
            @return: Threshold value
        """
        histogram = np.histogram(img, bins=self.max_pixel_val+1)
        thresholds = []
        for i in range(len(histogram[0])-1):
            thresholds.append(abs(sum(histogram[0][0:i]) - sum(histogram[0][i+1:])))
        return thresholds.index(min(thresholds))

    def BHT(self, img, fname = ''):
        """
            Balanced Histogram Thresholding
            @params: Input edge intensity array.
            @return: Binary Thresholded image
        """
        func_name = inspect.stack()[0][3]
        threshold = self.find_threshold(img)
        print("Threshold ",threshold)
        if threshold == 0:
            threshold = np.mean(img)
        print("Threshold ",threshold)
        w, h = img.shape
        out_im = np.zeros((w,h))
        for i in range(w):
            for j in range(h):
                out_im[i][j] = self.min_pixel_val if img[i][j] < threshold else self.max_pixel_val
        if not fname == '':
            fname = fname+'_'+func_name
            self.save_image(out_im, fname)
        return out_im

    def morphological(self, img, op = None):
        """
            Basic Operation for Dilate/Erode
            @params: Input Thresholded image
            @return: Dilated Image
        """
        if op == 'dilate':
            centre, replace = 1, 0
        elif op == 'erode':
            centre, replace = 0, 1
        else:
            raise KeyError("Provide operation to perform - 'erode'/'dilate'.")

        max_pixel = np.max(img)
        img = img/self.max_pixel_val if max_pixel == 0 else img/max_pixel
        w,h = img.shape
        for i in range(w):
            for j in range(h):
                if img[i][j] == centre:
                    if i > 0 and img[i-1][j] == replace: img[i-1][j] = 2;
                    if j > 0 and img[i][j-1] == replace: img[i][j-1] = 2;
                    if i+1 < w and img[i+1][j] == replace: img[i+1][j] = 2;
                    if j+1 < h and img[i][j+1] == replace: img[i][j+1] = 2;

        for i in range(w):
            for j in range(h):
                if img[i][j] == 2:
                    img[i][j] = centre;

        return img*self.max_pixel_val

    def dilate(self, img, fname = ''):
        """
            Dilate Edges
            @params: Input Thresholded image
            @return: Dilated Image
        """
        func_name = inspect.stack()[0][3]
        out = self.morphological(img, op='dilate')
        if not fname == '':
            fname = fname+'_'+func_name
            self.save_image(out, fname)
        return out

    def erode(self, img, fname = ''):
        """
            Erode Edges
            @params: Input Thresholded image
            @return: Eroded Image
        """
        func_name = inspect.stack()[0][3]
        out = self.morphological(img, op='erode')
        if not fname == '':
            fname = fname+'_'+func_name
            self.save_image(out, fname)
        return out

    def open_edge(self, img, fname = ''):
        """
            Open image edges
            @params: Input Thresholded image
            @return: Image with edges opened up
        """
        func_name = inspect.stack()[0][3]
        out = self.dilate(self.erode(img))
        if not fname == '':
            fname = fname+'_'+func_name
            self.save_image(out, fname)
        return out

    def close_edge(self, img, fname = ''):
        """
            Close image edges
            @params: Input Thresholded image
            @return: Image with edges closed together
        """
        func_name = inspect.stack()[0][3]
        out = self.erode(self.dilate(img))
        if not fname == '':
            fname = fname+'_'+func_name
            self.save_image(out, fname)
        return out

    def rgb2grayscale(self, im):
        """
            Converts RGB to Grayscale.
            @params: Input RGB image.
            @return: Grayscale image(1 channel)
        """
        if  len(im.shape) > 2:
            if im.shape[2] == 3: # Convert RGB image to Grayscale
                r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
                grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
                return grayscale
        else:
            return im
    def post_process(self, out, outfile):
        if not out.max() == 0:
            out *= 255.0/out.max()
        self.save_image(out, outfile)

        if self.BHT_enable: # Apply Balanced Histogram Thresholding
            out_bht = self.BHT(out, fname = outfile)

        if self.dilate_enable:
            self.dilate(out_bht, fname = outfile)

        if self.erode_enable:
            self.erode(out_bht, fname = outfile)

        if self.open_edges_enable:
            self.open_edge(out_bht, fname = outfile)

        if self.close_edges_enable:
            self.close_edge(out_bht, fname = outfile)

        if self.display_output: # Display output
            self.display(out)


    def process(self, img, imfilter, outfile):
        """
            Process image with provided filter.
            @params: Image pixel array.
            @return: Edge detection map
        """
        im = Image.open(img)
        arr = np.array(im)
        arr = self.rgb2grayscale(arr)

        if self.apply_gaussian: # Apply Gaussian Smoothening
           arr = self.conv(arr, self.gaussian_kernel)

        out = self.conv(arr, imfilter) # Apply the provided filter
        print("File {0} max val {1} min val {2}".format(outfile, np.max(out), np.min(out)))
        out = out.clip(min=0)

        #out *= 255.0/out.max()
        #out = np.asarray(out, np.uint8)
        self.post_process(out, outfile)

        return out

    def process_hor_vert_edges(self, horizontal_edge, vertical_edge, outfile):
        edge_intensity = np.sqrt(np.square(horizontal_edge)+np.square(vertical_edge))
        #edge_intensity = np.asarray(edge_intensity, np.uint8)
        edge_direction = np.arctan2(vertical_edge, horizontal_edge)/np.pi*180
        edge_intensity *= 255.0/edge_intensity.max()
        self.post_process(edge_intensity, outfile)
        return edge_intensity, edge_direction

    def forward(self, img, n = 10):
        """
            Forward edge filter
            @params: Input image.
            @return: Edge intensity map
        """
        self.apply_gaussian = False
        func_name = inspect.stack()[0][3]
        outfile =os.path.join(self.output_dir,img.split('/')[1].split('.')[0]+'_'+func_name)
        imfilter = np.transpose(np.array([[0, 0, 0],[0, n, 0],[0, -n, 0]]))
        return self.process(img, imfilter, outfile)

    def backward(self, img, n = 10):
        """
            Backward edge filter
            @params: Input image.
            @return: Edge intensity map
        """
        self.apply_gaussian = False
        func_name = inspect.stack()[0][3]
        outfile =os.path.join(self.output_dir,img.split('/')[1].split('.')[0]+'_'+func_name)
        imfilter = np.transpose(np.array([[0, -n, 0],[0, n, 0],[0, 0, 0]]))
        return self.process(img, imfilter, outfile)

    def central_finite_diff(self, img, n = 10):
        """
            Central Finite Differnce edge filter
            @params: Input image.
            @return: Edge intensity map
        """
        self.apply_gaussian = False
        func_name = inspect.stack()[0][3]
        outfile =os.path.join(self.output_dir,img.split('/')[1].split('.')[0]+'_'+func_name)
        imfilter = np.transpose(np.array([[0, -n, 0],[0, 0, 0],[0, n, 0]]))
        return self.process(img, imfilter, outfile)

    def sobel(self, img):
        """
            Sobel edge filter
            @params: Input image.
            @return: Edge intensity map
        """
        func_name = inspect.stack()[0][3]
        outfile =os.path.join(self.output_dir,img.split('/')[1].split('.')[0]+'_'+func_name)
        gx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        gy =  np.transpose(gx)
        im = Image.open(img)
        arr = np.array(im)
        arr = self.rgb2grayscale(arr)
        horizontal_edge = self.conv(arr, gx)
        vertical_edge = self.conv(arr, gy)
        return self.process_hor_vert_edges(horizontal_edge, vertical_edge, outfile)

    def prewitt(self, img):
        """
            Prewitt edge filter
            @params: Input image.
            @return: Edge intensity map
        """
        self.apply_gaussian = False
        func_name = inspect.stack()[0][3]
        outfile =os.path.join(self.output_dir,img.split('/')[1].split('.')[0]+'_'+func_name)
        gx = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
        gy =  np.transpose(gx)
        im = Image.open(img)
        arr = np.array(im)
        arr = self.rgb2grayscale(arr)
        horizontal_edge = self.conv(arr, gx)
        vertical_edge = self.conv(arr, gy)
        return self.process_hor_vert_edges(horizontal_edge, vertical_edge, outfile)

    def canny(self, img):
        """
            Canny edge filter
            @params: Input image.
            @return: Edge intensity map, direction map
        """
        func_name = inspect.stack()[0][3]
        outfile =os.path.join(self.output_dir,img.split('/')[1].split('.')[0]+'_'+func_name)
        # Apply Sobel Operator
        gx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        gy =  np.transpose(gx)
        im = Image.open(img)
        arr = np.array(im)
        arr = self.rgb2grayscale(arr)
        arr = self.conv(arr, gauss2D(shape=(3, 3), sigma=1.0)) # Apply Gaussian Filter
        horizontal_edge = self.conv(arr, gx)
        vertical_edge = self.conv(arr, gy)
        mag = np.sqrt(np.square(horizontal_edge)+np.square(vertical_edge))
        mag = np.asarray(mag, np.uint8)
        theta = np.arctan2(vertical_edge, horizontal_edge)/np.pi*180
        highThreshold = 91
        lowThreshold = 31
        non_max_sup = non_max_suppression(mag, theta)
        out = hysteresis(non_max_sup, mag)

        if self.BHT_enable: # Apply Balanced Histogram Thresholding
            out_bht = self.BHT(out)

        if self.dilate_enable:
            self.dilate(out_bht)

        if self.erode_enable:
            self.erode(out_bht)

        if self.display_output: # Display output
            self.display(out)

        self.display(out_bht)
        return out


    def laplacian(self, img):
        """
            Laplacian edge filter
            @params: Input image.
            @return: Edge intensity map
        """
        func_name = inspect.stack()[0][3]
        outfile =os.path.join(self.output_dir,img.split('/')[1].split('.')[0]+'_'+func_name)
        imfilter = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])
        return self.process(img, imfilter, outfile)

    def LoG(self, img, shape=(5,5), sigma=1.0):
        """
            Laplacian of Gaussian edge filter
            @params: Input image.
            @return: Edge intensity map
        """
        func_name = inspect.stack()[0][3]
        outfile =os.path.join(self.output_dir,img.split('/')[1].split('.')[0]+'_'+func_name)
        outfile = outfile+str(shape[0])+'x'+str(shape[1])+'_'+'sigma'+'_'+str(sigma).replace('.','_')
        #self.apply_gaussian = False
        return self.process(img, LoGKernel(shape=shape, sigma=sigma), outfile)

    def DoG(self, img, shape=(5,5), sigma1=1.0, sigma2=1.6):
        """
            Laplacian of Gaussian edge filter
            @params: Input image.
            @return: Edge intensity map
        """
        func_name = inspect.stack()[0][3]
        outfile =os.path.join(self.output_dir,img.split('/')[1].split('.')[0]+'_'+func_name)
        outfile = outfile+str(shape[0])+'x'+str(shape[1])+'_'+'sigma1'+'_'+str(sigma1).replace('.','_')+'sigma2'+'_'+str(sigma2).replace('.','_')
        #self.apply_gaussian = False
        print(DoGKernel(shape=shape, sigma1=sigma1, sigma2=sigma2))
        return self.process(img, DoGKernel(shape=shape, sigma1=sigma1, sigma2=sigma2), outfile)


if __name__ == "__main__":
    input_dir = "input"
    input_images = os.listdir("input")
    edge_detector = EdgeDetector()
    LoG_params = [((5,5), 0.6),((5,5), 0.7), ((5,5), 0.8), ((5,5), 0.9), ((5,5), 1.0), ((5,5), 1.1)]
    for img in input_images:
        input_file = os.path.join(input_dir, img)
        #edge_detector.forward(input_file)
        #edge_detector.backward(input_file)
        #edge_detector.sobel(input_file)
        #edge_detector.prewitt(input_file)
        edge_detector.canny(input_file)
        #edge_detector.laplacian(input_file)
        #edge_detector.LoG(input_file)
        #for params in LoG_params:
        #    shape, sigma = params
        #    edge_detector.LoG(input_file, shape=shape, sigma=sigma)
        #edge_detector.DoG(input_file)
