import cv2
import numpy  as np
from scipy.ndimage.filters import gaussian_filter
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers
model = cv2.dnn.readNet('pose_deploy.prototxt', 'pose_iter_116000.caffemodel')


def find_peaks(heatmap_avg, thre=0.1, sigma=3):
    all_peaks = []
    peak_counter = 0

    for part in range(0, heatmap_avg.shape[-1]):
        map_ori = heatmap_avg[:,:,part]
        map = gaussian_filter(map_ori, sigma=sigma)
        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]

        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > thre))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
    return all_peaks, peak_counter

vc = cv2.VideoCapture(0)

height = 224
width = 224
net_out = np.zeros((71,28,28))
cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
while True:
    _, img= vc.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    if (len(faces) == 0):
        continue
    x,y,w,h = faces[0]
    
    img = img[y:y+h, x:x+w]
    out_img = cv2.resize(img,(224,224))
    cv2.imshow('frame', img)
    cv2.waitKey(5)
    blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (height, width),
                              (0, 0, 0), swapRB=False, crop=False)
    model.setInput(blob)
    outs = model.forward()
    points = []
    for i in range(71):
        hetMap = outs[0, i, :, :]

        _, conf, _, point = cv2.minMaxLoc(hetMap)
        x = (width * point[0]) / outs.shape[3]
        y = (height * point[1]) / outs.shape[2]
        print (x)
        print (y)
        if (conf > 0.1 ):
            cv2.circle(out_img, (x,y),5,(10*i,1*i,15*i))
 #           point.append((x,y))
        cv2.imshow('out_img', out_img)
        cv2.imshow('heatmap', hetMap)
        cv2.waitKey()
