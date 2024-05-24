import numpy as np
import cv2
import time
from vpu_model import VPUModel
from scipy.special import softmax

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
# blob_file = work_dir + "resnet50v1_50_ov2023_fp16_nchw_sp50_bgr_incpreproc_cpu_defaultquantization_statistics.blob"
blob_file = work_dir + "resnet50v1.blob"
input_name = 'images'
img_width = 224
device = "NPU"

my_img = "test_img/african_bush_elephant.jpg" 

img_org = cv2.imread(my_img)

img = cv2.resize(img_org,(img_width,img_width))  # out of resize is bgr

#: could not broadcast input array from shape (224,224,3) into shape (1,3,224,224)
data = np.array(img)
data = np.transpose(data, (2, 0, 1))   # hwc ->  chw  
                                        # 012     201 
data = data.reshape(1,3,img_width,img_width)   # 3,224,224 -> 1,3,224,224
data = np.float16(data)

inference_requests = 1
inference_runs = 1
input_data = {input_name:data}
model = VPUModel(blob_file,device,inference_requests)
inference_count = inference_runs*inference_requests
start_time = time.time()
for ir in range(0,inference_runs):
    model.run(input_data)
end_time = time.time()

profiling = model.get_profiling()

fps_str = f'Performance for {inference_count:d} inferences: {(inference_count / (end_time - start_time)):2.1f} FPS'
print(fps_str)
result = model.get_output()

sm = softmax(result[0])

k = 5
top_k = topk_by_partition(sm,k,1,False)


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
cv2.imshow(my_img, img_org)
cv2.waitKey(0)

