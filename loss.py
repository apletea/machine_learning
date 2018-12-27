def get_model_best_cl(num_classes):
    vec_dim = 128
    model = load_model('models/{}.hdf5'.format('inference_Xception_1.2_remote_kkk'), custom_objects={'tf':tf})
    last = model.layers[-5].output
    last = Dense(vec_dim)(last)
#    last = Lambda(lambda x: K.tf.nn.softmax(x))(last)
#Center loss
    ip1 = model.get_layer('ip1').output
    ip2 = Dense(num_classes)(ip1) 
    ip2 = Activation('softmax')(ip2)
    emb_inputs = []
    for i in range(num_classes):
        emb_inputs.append(Input(shape=(1,), name='input_{}'.format(str(i+6))))

    center_model_iput = Input(shape=(1,), name='input_2') 
    centers = Embedding(num_classes,vec_dim)(center_model_iput)
    centers = Model(center_model_iput, centers)
    centers_arr = []
    for i in range(num_classes):
        centers_arr.append(centers(emb_inputs[i]))
    l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0])+1/(K.sum(K.square([x[0]-x[i][:,0] for i in range(2,len(x))]))+1),keepdims=True),name='l2_loss')([ip1] + centers_arr)
    model_centerloss = Model(inputs=[model.input] + emb_inputs, outputs=[ip2,l2_loss]) 
#   for index in range(len(model.layers)-7):
#       model_centerloss.layers[index].trainable=False
    return model_centerloss
