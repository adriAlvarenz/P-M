import cv2
from skimage import io
F_PATH = "D:\SCHOOL\CUARTO AÑO\Tesis\+PYDENS\\fastaniso-master\\fastaniso_master\\fastaniso_output.jpg"
P_PATH= "D:\SCHOOL\CUARTO AÑO\Tesis\+PYDENS\pydens\\tutorials\\hope.jpg"
O_PATH = "D:\SCHOOL\CUARTO AÑO\Tesis\+PYDENS\pydens\\tutorials\\initial_image.jpg"

# fastaniso_img = io.imread(F_PATH)
# pydens_img = io.imread(P_PATH)
# original_img = io.imread(O_PATH)
b = [10,20,30,40,50,60,1000,3000,4000]
def apply_canny():
    a = 'pydens_edges'
    for i in range(0,10):
        img = io.imread('mod_t='+str(b[i])+'.png')
        edge_img = edge_detection(img)
        io.imsave('mod_t'+str(b[i])+'canny.png',edge_img)

def edge_detection(image):
    return cv2.Canny(image, 100, 200) 
 
apply_canny()

# # Applying the Canny Edge filter
# f_img = edge_detection(fastaniso_img)
# p_img = edge_detection(pydens_img )
# o_img = edge_detection(original_img)

# io.imsave("o_output.jpg", o_img)
# io.imsave("f_output.jpg", f_img)
# io.imsave("p_output.jpg", p_img)



  

