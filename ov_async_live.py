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
  
# define a video capture object
vid = cv2.VideoCapture(0)

work_dir = "model/"
xml_file = work_dir + "ResNet-50-model.xml"
bin_file = work_dir + "ResNet-50-model.bin"
input_name = 'data'
img_width = 224
device = "CPU"

max_num_requests = 4

ie = IECore()
net = ie.read_network(model=xml_file, weights=bin_file)
exec_net = ie.load_network(net, device, num_requests=max_num_requests)
request = exec_net.requests[0]

request_frame = []

# Define a callback function that will be called when the request is complete
def callback(request_id):    
    frame = request_frame[request_id]

    # Get the output of the model
    result = exec_net.requests[request_id].output_blobs['prob'].buffer
        
    k = 5
    top_k = topk_by_partition(result,k,1,False)


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
        #print('%.1f%%\t%s' % (probs[i]*100,imgnet.labels[classes[i]]))
        result = f'{probs[i]*100:2.1f}% {imgnet.labels[classes[i]]}'
        cv2.putText(frame,result, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
        text_y = text_y + 30
        bottomLeftCornerOfText = (50,text_y)

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



    img_org = frame
    org_width = img_org.shape[1]
    org_height = img_org.shape[0]

    img = cv2.resize(img_org,(img_width,img_width))  # out of resize is bgr

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
        callback(request_slot)

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