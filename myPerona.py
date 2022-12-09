import fastaniso
from scipy import misc
import imageio
import matplotlib.pyplot as plt
from skimage import io
import cv2
import numpy as np
import os
#import generateDS.my_globals as globals
import pathlib
from canny import * 
from PIL import Image as im


# img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),option=1,ploton=False):
#photo =   fastaniso.anisodiff()
#f = misc.face()
#imageio.imsave('dog.1.jpg', f)
# plt.imshow(f)
# plt.show()


#image = cv2.imread("Boob.jpg")
# image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#Convertir imagen a imagen en escala de grises
# cv2.imshow("Original",image)
# cv2.waitKey()

# Detección de borde de Laplace
# lap = cv2.Canny(image,100,200)# Detección de borde de Laplace
# lap = np.uint8(np.absolute(lap))## Ir al valor absoluto de vuelta
# cv2.imshow("Laplacian",lap)
# cv2.imwrite('Bordes.png',lap)


# multiple images
rute_up = "D:\SCHOOL\CUARTO AÑO\Tesis\+PYDENS\fastaniso-master\ds_ini"
rute_down = "D:\SCHOOL\CUARTO AÑO\Tesis\+PYDENS\fastaniso-master\ds_final"
# rute_salt_and_pepper = "c:\\Users\Bell\\Documents\\GitHub\\Addin-noise-to-images\\diferent_noise\\salt_and_pepper"

print(str(pathlib.Path(__file__).parent.absolute()))
number_of_iterations = 1000


def image_retrieve():
    rootDir = rute_up
    rootFinal = rute_down
    i = 0
    for dirName, subdirList, fileList in os.walk(rootDir):

        for fname in fileList:
            # creamos la carpeta asociada a la foto
            if not (os.path.isdir(str(rootFinal)+"\\"+str(fname)[:len(str(fname))-4] + str('_') + str(i))):
                os.mkdir(str(rootFinal)+"\\"+str(fname)
                         [:len(str(fname))-4] + str('_') + str(i))
            # aca cuando metamos concurrencia
            # concurrency_handler()
            print(dirName)
            print("-----------------------")
            print(subdirList)
            print("-----------------------")
            print(fileList)
            print("-----------------------")
            print(fname)
            print("-----------------------")
            result_list = []
            img_PIL = io.imread(str(dirName)+'\\'+str(fname))

            j = 0
            os.chdir(rute_down+'\\'+str(fname)
                     [:len(str(fname))-4] + str('_') + str(i))
            io.imsave(str(i)+'_ite_'+str(0)+'.jpg', img_PIL)

            fastaniso.anisodiff(img_PIL, number_of_iterations, result_list)
            for item in result_list:
                j += 1
                io.imsave(str(i)+'_ite_'+str(j)+'.jpg', result_list[j-1])

            i += 1


image_retrieve()

def cart_prod(*arrs):
    grids = np.meshgrid(*arrs, indexing='ij')
    return np.stack(grids, axis=-1).reshape(-1, len(arrs))

# def f(x,y):
#     z = (x - 1)**2 + (y - 1.5)**2
#     for i in range(0,len(z)):
#         if z[i] <= 1:
#             z[i] = 0.5
#         else: z[i] = 1

#     return z
def f(x,y):
    z = (x - .5)**2 + (y - .5)**2
    for i in range(0,len(z)):
        if x[i]==0 or y[i]==0 or x[i]==1 or y[i]==1:
            z[i] = 0
            continue
        if z[i] <= 0.03:
            z[i] = 0.5
        else: z[i] = 1

    return z
vals = np.linspace(0,1,50)
grid = cart_prod(vals,vals)
xs, ys = grid[:, 0:1], grid[:, 1:2]

matt = f(xs,ys).reshape(50,50)
io.imsave("cc.jpg",matt)
# print(matt)
# img_PIL = io.imread('hope.jpg')
img_PIL = io.imread('tttt.jpg')
# initial_img = cv2.resize(img_PIL,(200,200),interpolation = cv2.INTER_AREA)
for i in range(0,10):
    ani = fastaniso.anisodiff(matt, kappa=50,niter=i)
    io.imsave('fastaniso_output' + str(i) + '.jpg', ani)


# ani_edge = edge_detection(np.matrix(ani))
# io.imsave('edge.jpg', ani_edge)

# print(type(img_PIL))
# print(img_PIL.dtype)
# print(img_PIL.shape)
# print(img_PIL.min(), img_PIL.max())
# plt.imshow(img_PIL)
# # io.imsave('img.jpg', ani)
# print("a")
