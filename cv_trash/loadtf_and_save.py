import tensorflow as tf
from code import OpenPose as OP
from tensorflow.python.tools import freeze_graph

def load_graph(graph_name):
    with tf.gfile.GFile(graph_name, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:

        tf.import_graph_def(graph_def, name='')
    return graph


#graph = load_graph('out.pb')
#for op in graph.get_operations():
#    prinst (op.name)
data = tf.placeholder(tf.float32, [ 1,224,224,3])
net = OP({'image' : data})
print (net.get_output())
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    net.load('weights.bin', sess)

    tf.train.write_graph(sess.graph_def, '.', 'graph.pb',False)

#    freeze_graph.freeze_graph("face.pb", "",
#                                      True, "chc.chkc", 'fc8/fc8',
#                                      'save/restore_all', 'save/Const:0',
#                                      "face" + '_frozen.pb', False, "")


#    for op in net.get_operations():
#        print (op.name)
