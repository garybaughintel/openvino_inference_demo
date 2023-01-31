import numpy as np
from openvino.inference_engine import IENetwork, IECore, get_version as ie_get_version
import cv2
import time
import pdb
import coco80_labels as coco
import colour_palette as palette

np.set_printoptions(suppress=True, precision=3)

def clip_detection(box, size):
  box[0] = max(int(box[0]), 0) #ymin
  box[1] = max(int(box[1]), 0) #xmin  
  box[2] = min(int(box[2]), size[0]) #ymax
  box[3] = min(int(box[3]), size[1]) #xmax
  return np.int32(box)

def resize_image_letterbox(image, size, interpolation=cv2.INTER_LINEAR):
    ih, iw = image.shape[0:2]
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = cv2.resize(image, (nw, nh), interpolation=interpolation)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    resized_image = np.pad(image, ((dy, dy + (h - nh) % 2), (dx, dx + (w - nw) % 2), (0, 0)),
                           mode='constant', constant_values=0)
    return resized_image


work_dir = "model/public/yolo-v3-tiny-onnx/FP32/"
xml_file = work_dir + "yolo-v3-tiny-onnx.xml"
bin_file = work_dir + "yolo-v3-tiny-onnx.bin"
input_name = 'input_1'
img_width = 416
device = "CPU"

my_img = "test_img/dog.jpg"

img_org = cv2.imread(my_img)
img_size = [img_org.shape[0], img_org.shape[1]]
# img = cv2.resize(img_org,(img_width,img_width))  # out of resize is bgr
img = resize_image_letterbox(img_org,[img_width,img_width],2)

#: could not broadcast input array from shape (416,416,3) into shape (1,3,416,416)
data = np.array(img)
data = np.transpose(data, (2, 0, 1))   # hwc ->  chw  
                                        # 012     201 
data = data.reshape(1,3,img_width,img_width)   # 3,416,416 -> 1,3,416,416
data = data.astype('float32')


input_data = {input_name:data,'image_shape':np.array([img_size], dtype=np.float32)}

ie = IECore()
net = ie.read_network(model=xml_file, weights=bin_file)
#net.add_outputs("yolo_evaluation_layer_1/concat_6:0_btc")
exec_net = ie.load_network(net, device)

print("Starting inference")
result = exec_net.infer(input_data)

boxes = result['yolonms_layer_1'][0]
scores = result['yolonms_layer_1:1'][0]
indices = result['yolonms_layer_1:2'][0]

colours = palette.get_colour_palette(len(coco.labels))
confidence_threshold = 0.5
label_height = 25

for index in indices :
  if(index[0] == -1) :
    break

  score = scores[tuple(index[1:])]

  if(score >= confidence_threshold) :
    # ymin, xmin, ymax, xmax
    box = boxes[index[2]]
    box = clip_detection(box, img_size)      
    label_index = index[1]
    ymin = box[0]
    xmin = box[1]
    ymax = box[2] 
    xmax = box[3]
    print('%.1f%%\t%s' % (score*100,coco.labels[label_index]))
    det_label = f'{score*100:2.1f}% {coco.labels[label_index]}'
    
    colour = colours[label_index]
    text_colour = (28,28,28)

    cv2.rectangle(img_org, (xmin, ymin), (xmax, ymax), colour, 2)
    cv2.rectangle(img_org, (xmin, ymin-label_height), (xmax, ymin), colour, -1)
    cv2.rectangle(img_org, (xmin, ymin-label_height), (xmax, ymin), colour, 2)
    cv2.putText(img_org, det_label,
                (xmin, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_colour, 1)

cv2.imwrite('test_img/yolo_out.jpg', img_org)
cv2.imshow("input", img_org)
cv2.waitKey(0)









