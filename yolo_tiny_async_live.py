import numpy as np
from openvino.inference_engine import IENetwork, IECore, get_version as ie_get_version
import cv2
import time
import coco80_labels as coco
import colour_palette as palette

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
  
# define a video capture object
# vid = cv2.VideoCapture(0)
vid = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

work_dir = "model/public/yolo-v3-tiny-onnx/FP32/"
xml_file = work_dir + "yolo-v3-tiny-onnx.xml"
bin_file = work_dir + "yolo-v3-tiny-onnx.bin"
input_name = 'input_1'
img_width = 416
device = "CPU"

max_num_requests = 4

ie = IECore()
net = ie.read_network(model=xml_file, weights=bin_file)
exec_net = ie.load_network(net, device, num_requests=max_num_requests)
request = exec_net.requests[0]

request_frame = []
total_inference_frames = 0
start_time = time.time()
fps_str = ''

colours = palette.get_colour_palette(len(coco.labels))
confidence_threshold = 0.3
label_height = 25
lineType = 2

# Define a callback function that will be called when the request is complete
def inference_result_ready(request_id):  
    global total_inference_frames 
    global fps_str 
    global start_time
    frame = request_frame[request_id]
    img_size = [frame.shape[0], frame.shape[1]]
    # Get the output of the model
    boxes = exec_net.requests[request_id].output_blobs['yolonms_layer_1'].buffer[0]
    scores = exec_net.requests[request_id].output_blobs['yolonms_layer_1:1'].buffer[0]  
    indices = exec_net.requests[request_id].output_blobs['yolonms_layer_1:2'].buffer[0] 
   

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
            det_label = f'{score*100:2.1f}% {coco.labels[label_index]}'
            
            colour = colours[label_index]
            text_colour = (28,28,28)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), colour, 2)
            cv2.rectangle(frame, (xmin, ymin-label_height), (xmax, ymin), colour, -1)
            cv2.rectangle(frame, (xmin, ymin-label_height), (xmax, ymin), colour, 2)
            cv2.putText(frame, det_label,
                        (xmin, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_colour, 1)

    total_inference_frames += 1

    if total_inference_frames == 30:
        end_time = time.time()
        fps_str = f'{(30 / (end_time - start_time)):2.1f} FPS'        
        total_inference_frames = 0
        start_time = time.time()

    cv2.putText(frame,fps_str, 
        (frame.shape[1] - 150,frame.shape[0] - 50), 
        cv2.FONT_HERSHEY_PLAIN, 
        1.8,
        (0,255,0),
        1,
        lineType)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    



    

print("Starting inference")

request_slot = 0
get_inference_result = False

while(True):      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()   

    if len(request_frame) < max_num_requests :
        request_frame.append(frame)
    else :
        get_inference_result = True
        request_frame[request_slot] = frame
    
    img_size = [frame.shape[0], frame.shape[1]]
    img = resize_image_letterbox(frame,[img_width,img_width],2)

    #: could not broadcast input array from shape (224,224,3) into shape (1,3,224,224)
    data = np.array(img)
    data = np.transpose(data, (2, 0, 1))   # hwc ->  chw  
                                            # 012     201 
    data = data.reshape(1,3,img_width,img_width)   # 3,224,224 -> 1,3,224,224
    data = data.astype('float')

    input_data = {input_name:data,'image_shape':np.array([img_size], dtype=np.float32)}

    # Start an asynchronous inference request    
    exec_net.requests[request_slot].wait()
    
    if get_inference_result == True:
        inference_result_ready(request_slot)

    exec_net.requests[request_slot].async_infer(input_data)    
    
    request_slot = request_slot + 1
    request_slot = request_slot % max_num_requests
      
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()