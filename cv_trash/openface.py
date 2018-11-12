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
#        plt.imshow(map)
#        plt.show()

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
while True:
    _, img= vc.read()
    cv2.imshow('frame', img)
    cv2.waitKey(5)
    blob = cv2.dnn.blobFromImage(img,1/255, (224,224), (0,0,0), True, crop=False)
    model.setInput(blob)
    outs = model.forward(get_output_layers(model))
    net_out = outs[0][0]
    print net_out.shape
    reszied_out = np.transpose(net_out, (1,2,0))
    print reszied_out.shape
    reszied_out = cv2.resize(reszied_out, (height, width))
    print reszied_out.shape
    peakss = find_peaks(reszied_out, 0.1, 3)[0]
    print peakss
    for peak in peakss:
        if (len(peak)):
            peak = peak[0]
            print 'darw marker'
            cv2.drawMarker(img,(peak[0], peak[1]), (0, 255,0), cv2.MARKER_STAR)
    img = np.clip(img, 0 ,255).astype(np.uint8)
    cv2.imshow('res',img)
    cv2.waitKey(5)
