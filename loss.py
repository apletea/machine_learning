def get_model_best_cl(num_classes):
    vec_dim = 2
    model = load_model('models/{}.hdf5'.format('inference_Xception_1.2_remote_kkk'), custom_objects={'tf':tf})
    last = model.layers[-5].output
    last = Dense(vec_dim)(last)
#    last = Lambda(lambda x: K.tf.nn.softmax(x))(last)
#Center loss
    ip1 = model.get_layer('ip1').output
    ip1 = Dense(vec_dim)(ip1)
    ip1 = PReLU()(ip1)
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
     
    inverse_loss = Lambda(lambda x : 1/K.sum(K.square([x[0] - x[i][:,0] for i in range(2,len(x))])+1,keepdims=True))([ip1] + centers_arr) 
    inverse_loss = Reshape((-1,1))(inverse_loss)
    l2_loss = Lambda(lambda x:K.mean(K.sum(K.square(x[0]-x[1][:,0]) + x[len(x) -1][0],keepdims=True,axis=(1)),axis=0, keepdims=True),name='l2_loss')([ip1] + centers_arr + [inverse_loss])
#    inverse_loss = Lambda(lambda x : 1/K.sum(K.square([x[0] - x[i][:,0] for i in range(2,len(x))])+1,keepdims=True, axis=(0,2)))([ip1] + centers_arr) 
#    inverse_loss = Reshape((-1,1))(inverse_loss)
#    l2_loss = Lambda(lambda x:K.mean(K.sum(K.square(x[0]-x[1][:,0]) + x[len(x) -1][0],keepdims=True,axis=(1)),axis=0, keepdims=True),name='l2_loss')([ip1] + centers_arr + [inverse_loss])
    model_centerloss = Model(inputs=[model.input] + emb_inputs, outputs=[ip2,l2_loss]) 
#   for index in range(len(model.layers)-7):
#       model_centerloss.layers[index].trainable=False
    return model_centerloss
