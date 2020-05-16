

```python
import torch
import cv2
import numpy as np
from matplotlib import pyplot
```


```python
img = cv2.imread('lenna.png')
img = cv2.imread('pycharm.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pyplot.imshow(img)
```








![png](images/output_1_1.png)



```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
pyplot.imshow(img, cmap="gray")
```








![png](images/output_2_1.png)



```python

img = torch.from_numpy(img.reshape(1, 1, *img.shape) / 255.)
img = img.type(torch.FloatTensor)
print(img.shape)
```

    torch.Size([1, 1, 32, 32])



```python
def convolve(img, kernel):
    kernel = torch.tensor(kernel)
    kernel = kernel.type(torch.FloatTensor)
    kernel = kernel.view(1, 1, *kernel.shape).repeat(1, 1, 1, 1)
    return torch.conv2d(img, kernel, padding=0)
```


```python
# Identity
convolved = convolve(img,
                     [
                         [0., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 0.]
                     ]
                    )
pyplot.imshow(convolved[0, 0], cmap="gray")
```








![png](images/output_5_1.png)



```python
# Edge detection 2 (Laplace 8)
convolved = convolve(img,
                     [
                         [-1., -1., -1.],
                         [-1., 8., -1.],
                         [-1., -1., -1.]
                     ]
                    )
pyplot.imshow(convolved[0, 0], cmap="gray")
```








![png](images/output_6_1.png)



```python
# Edge detection 2 (Prewitt vertical)
prewitt_v = convolve(img,
                     [
                         [1., 0., -1.],
                         [1., 0., -1.],
                         [1., 0., -1.]
                     ]
                    )
pyplot.imshow(prewitt_v[0, 0], cmap="gray")
```








![png](images/output_7_1.png)



```python
# Edge detection 2 (Prewitt hotizontal)
prewitt_h = convolve(img,
                     [
                         [1., 1., 1.],
                         [0., 0., 0.],
                         [-1., -1., -1.]
                     ]
                    )
pyplot.imshow(prewitt_h[0, 0], cmap="gray")
```








![png](images/output_8_1.png)



```python
prewitt_combined = pow(pow(prewitt_h, 2) + pow(prewitt_v, 2), 0.5)
pyplot.imshow(prewitt_combined[0, 0], cmap="gray")
```








![png](images/output_9_1.png)



```python
# Edge detection 2 (Sobel vertical)
sobel_v = convolve(img,
                     [
                         [1., 0., -1.],
                         [2., 0., -2.],
                         [1., 0., -1.]
                     ]
                    )
pyplot.imshow(sobel_v[0, 0], cmap="gray")
```








![png](images/output_10_1.png)



```python
# Edge detection 2 (Sobel hotizontal)
sobel_h = convolve(img,
                     [
                         [1., 2., 1.],
                         [0., 0., 0.],
                         [-1., -2., -1.]
                     ]
                    )
pyplot.imshow(sobel_h[0, 0], cmap="gray")
```








![png](images/output_11_1.png)



```python
sobel_combined = pow(pow(sobel_h, 2) + pow(sobel_v, 2), 0.5)
pyplot.imshow(sobel_combined[0, 0], cmap="gray")
```








![png](images/output_12_1.png)



```python
# Sharpen 1
convolved = convolve(img,
                     [
                         [-1., -1., -1.],
                         [-1., 9., -1.],
                         [-1., -1., -1.]
                     ]
                    )
pyplot.imshow(convolved[0, 0], cmap="gray")
```








![png](images/output_13_1.png)



```python
# Sharpen 2
convolved = convolve(img,
                     [
                         [0., -1., 0.],
                         [-1., 5., -1.],
                         [0., -1., 0.]
                     ]
                    )
pyplot.imshow(convolved[0, 0], cmap="gray")
```








![png](images/output_14_1.png)



```python
# Emboss
convolved = convolve(img,
                     [
                         [-2., -1., -1.],
                         [-1., 1., 1.],
                         [0., 1., 2.]
                     ]
                    )
pyplot.imshow(convolved[0, 0], cmap="gray")
```








![png](images/output_15_1.png)



```python
# Box blur
convolved = convolve(img,
                     [
                         [1., 1., 1.],
                         [1., 1., 1.],
                         [1., 1., 1.]
                     ]
                    )
pyplot.imshow(convolved[0, 0], cmap="gray")
```








![png](images/output_16_1.png)



```python
# Gaussian blur
convolved = convolve(img,
                     [
                         [1., 2., 1.],
                         [2., 4., 2.],
                         [1., 2., 1.]
                     ]
                    )
pyplot.imshow(convolved[0, 0], cmap="gray")
```








![png](images/output_17_1.png)



```python
# a little more complex kernel cropped from the image
kernel = cv2.imread("C.png")
kernel = cv2.cvtColor(kernel, cv2.COLOR_BGR2GRAY)
pyplot.imshow(kernel, cmap="gray")
```








![png](images/output_18_1.png)



```python
convolved = convolve(img, kernel)
# convolved = torch.max_pool2d(convolved, 3)
pyplot.imshow(convolved[0, 0], cmap="gray")
```








![png](images/output_19_1.png)



```python

kernel = torch.tensor(kernel)
kernel = kernel.view(1, 1, *kernel.shape).repeat(1, 1, 1, 1)
kernel = kernel.type(torch.FloatTensor)
transpose = torch.conv_transpose2d(convolved, kernel)

pyplot.imshow(transpose[0, 0], cmap="gray")

```








![png](images/output_20_1.png)



```python
pyplot.imshow(img[0, 0], cmap="gray")
```








![png](images/output_21_1.png)



```python

```
