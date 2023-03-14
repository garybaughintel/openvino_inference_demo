import numpy as np
from openvino.inference_engine import IENetwork, IECore, get_version as ie_get_version
import cv2

import coco91_labels as coco
import colour_palette as palette

np.set_printoptions(suppress=True, precision=3)

def clip_detection(box, size):
  # x_min, y_min, x_max, y_max
  box[0] =  box[0] * size[1]
  box[2] =  box[2] * size[1]
  box[1] =  box[1] * size[0]
  box[3] =  box[3] * size[0]
  box[0] = max(int(box[0]), 0) #xmin
  box[1] = max(int(box[1]), 0) #ymin  
  box[2] = min(int(box[2]), size[1]) #xmax
  box[3] = min(int(box[3]), size[0]) #ymax
  return np.int32(box)



work_dir = "model/public/ssdlite_mobilenet_v2/FP32/"
xml_file = work_dir + "ssdlite_mobilenet_v2.xml"
bin_file = work_dir + "ssdlite_mobilenet_v2.bin"
input_name = 'image_tensor'
img_width = 300
device = "CPU"

# my_img = "test_img/dog.jpg"
my_img = "test_img/cricket.jpg"
# my_img = "test_img/broccoli-orange.jpg"
#my_img = "test_img/african_bush_elephant.jpg"
#my_img = "test_img/fruits-and-vegetables.jpg"

img_org = cv2.imread(my_img)
img_size = [img_org.shape[0], img_org.shape[1]]
img = cv2.resize(img_org,(img_width,img_width))  # out of resize is bgr

cv2.imwrite('test_img/ssdlite_resize.jpg', img)
#: could not broadcast input array from shape (416,416,3) into shape (1,3,416,416)
data = np.array(img)
data = np.transpose(data, (2, 0, 1))   # hwc ->  chw  
                                        # 012     201 
data = data.reshape(1,3,img_width,img_width)   # 3,416,416 -> 1,3,416,416
data = data.astype('float32')


input_data = {input_name:data}

ie = IECore()
net = ie.read_network(model=xml_file, weights=bin_file)
exec_net = ie.load_network(net, device)

print("Starting inference")
result = exec_net.infer(input_data)
detection_output = result['DetectionOutput'][0][0]

colours = palette.get_colour_palette(len(coco.labels))
confidence_threshold = 0.3
label_height = 25

for detection in detection_output :
  score = np.float32(detection[2])
  if(score < confidence_threshold) :
    break

  box = detection[3:]
  label_index = np.int32(detection[1])

  if(score >= confidence_threshold) :
    # x_min, y_min, x_max, y_max    
    box = clip_detection(box, img_size)      
    
    ymin = box[1]
    xmin = box[0]
    ymax = box[3] 
    xmax = box[2]
    print('%.1f%%\t%s' % (score*100,coco.labels[label_index]))
    det_label = f'{score*100:2.1f}% {coco.labels[label_index]}'
    
    colour = colours[label_index]
    text_colour = (28,28,28)

    cv2.rectangle(img_org, (xmin, ymin), (xmax, ymax), colour, 2)
    cv2.rectangle(img_org, (xmin, ymin-label_height), (xmax, ymin), colour, -1)
    cv2.rectangle(img_org, (xmin, ymin-label_height), (xmax, ymin), colour, 2)
    cv2.putText(img_org, det_label,
                (xmin, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_colour, 1)

cv2.imwrite('test_img/ssdlite_out.jpg', img_org)
cv2.imshow(my_img, img_org)
cv2.waitKey(0)









