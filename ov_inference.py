import numpy as np
from openvino.inference_engine import IENetwork, IECore, get_version as ie_get_version
import cv2
import time
import pdb
import imagenet_labels as imgnet

def topk_by_partition(input, k, axis=None, ascending=True):
    if not ascending:
        input *= -1
    ind = np.argpartition(input, k, axis=axis)
    ind = np.take(ind, np.arange(k), axis=axis) # k non-sorted indices
    input = np.take_along_axis(input, ind, axis=axis) # k non-sorted values

    # sort within k elements
    ind_part = np.argsort(input, axis=axis)
    ind = np.take_along_axis(ind, ind_part, axis=axis)
    if not ascending:
        input *= -1
    val = np.take_along_axis(input, ind_part, axis=axis) 
    return ind, val

np.set_printoptions(suppress=True, precision=3)

work_dir = "model/"
xml_file = work_dir + "ResNet-50-model.xml"
bin_file = work_dir + "ResNet-50-model.bin"
input_name = 'data'
img_width = 224
device = "GPU"

my_img = "test_img/african_bush_elephant.jpg"

img_org = cv2.imread(my_img)

img = cv2.resize(img_org,(img_width,img_width))  # out of resize is bgr

#: could not broadcast input array from shape (224,224,3) into shape (1,3,224,224)
data = np.array(img)
data = np.transpose(data, (2, 0, 1))   # hwc ->  chw  
                                        # 012     201 
data = data.reshape(1,3,img_width,img_width)   # 3,224,224 -> 1,3,224,224
data = data.astype('float')

input_data = {input_name:data}

ie = IECore()
net = ie.read_network(model=xml_file, weights=bin_file)
exec_net = ie.load_network(net, device)

print("Starting inference")
result = exec_net.infer(input_data)


k = 5
top_k = topk_by_partition(result['prob'],k,1,False)


classes = top_k[0][0]
probs = top_k[1][0]
text_y = 50
text_x = 50
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (50,text_y)
fontScale              = 0.7
fontColor              = (255,0,255)
thickness              = 2
lineType               = 2

for i in range(0,k):
    print('%.1f%%\t%s' % (probs[i]*100,imgnet.labels[classes[i]]))
    result = f'{probs[i]*100:2.1f}% {imgnet.labels[classes[i]]}'
    cv2.putText(img_org,result, 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    text_y = text_y + 30
    bottomLeftCornerOfText = (50,text_y)



cv2.imwrite('test_img/resnet_' + device + '_out.jpg', img_org)
cv2.imshow("input", img_org)
cv2.waitKey(0)

