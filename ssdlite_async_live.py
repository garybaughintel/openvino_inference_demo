import numpy as np
from openvino.inference_engine import IENetwork, IECore, get_version as ie_get_version
import cv2
import time
import coco91_labels as coco
import colour_palette as palette

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
  
# define a video capture object
# vid = cv2.VideoCapture(0)
vid = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

work_dir = "model/public/ssdlite_mobilenet_v2/FP32/"
xml_file = work_dir + "ssdlite_mobilenet_v2.xml"
bin_file = work_dir + "ssdlite_mobilenet_v2.bin"
input_name = 'image_tensor'
img_width = 300
device = "GPU"

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
    result = exec_net.requests[request_id].output_blobs['DetectionOutput'].buffer
        
    detection_output = result[0][0]

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
    cv2.imshow('SSDLite MobileNetV2 (async)', frame)
    



    

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

   
    img = cv2.resize(frame,(img_width,img_width))  # out of resize is bgr

    #: could not broadcast input array from shape (224,224,3) into shape (1,3,224,224)
    data = np.array(img)
    data = np.transpose(data, (2, 0, 1))   # hwc ->  chw  
                                            # 012     201 
    data = data.reshape(1,3,img_width,img_width)   # 3,224,224 -> 1,3,224,224
    data = data.astype('float')

    input_data = {input_name:data}

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