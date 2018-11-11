import tensorflow as tf
import numpy as np
import cv2
from tensorflow.python.tools import freeze_graph

def load_graph(filename):
    with tf.gfile.GFile(filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:

        tf.import_graph_def(graph_def, name='')

    return graph


graph = load_graph('graph.pb')
for op in graph.get_operations():
    print (op.name)

#x_tensor = graph.get_tensor_by_name("conv1_1/conv1_1:0")
x_tensor = graph.get_tensor_by_name("Placeholder:0")
out = graph.get_tensor_by_name("Mconv7_stage6/BiasAdd:0") 
print (x_tensor)
print (out)
vc = cv2.VideoCapture(0)
input_img = np.zeros((1,224,224,3)) 
with graph.as_default():
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True :
            _,img = vc.read()
            img = cv2.resize(img,(224,224))
            input_img[0] = img
            input_img = np.array(input_img,np.float32)/ 255 
            cv2.imshow('frame', img)
            cv2.waitKey(5)
            res = sess.run(out, feed_dict={x_tensor : input_img})
            print (res)
    

