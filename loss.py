    ip1 = PReLU(name='ip1')(last)
    ip2 = Dense(num_classes)(ip1)
    ip2 = Activation('sigmoid')(ip2)
    lambda_c = 0.2
    emb_inputs = []
    for i in num_classes:
        emb_inputs.append(Input(shape=(1,), name='input_{}'.format(str(i+2))))
#        input_target      = Input(shape=(1,), name='input_3') # single value ground truth labels as inputs
#        input_target_inv  = Input(shape=(1,), name='input_4') 
    center_model_iput = Input(shape=(1,), name='input_5') 
    centers = Embedding(num_classes,vec_dim)(center_model_iput)
    centers = Model(center_model_iput, centers)
    centers_arr = []
    for i in num_classes:
        centers_arr.append(centers[emb_inputs[i]])
    cur_cen   = centers(input_target)
    other_cen = centers(input_target_inv)
    l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0])/(K.square(x[0]-x[2][:,0])+1),keepdims=True),name='l2_loss')([ip1, cur_cen, other_cen])
    model_centerloss = Model(inputs=[model.input,input_target,input_target_inv],outputs=[ip2,l2_loss]) 
    plot_model(model_centerloss, to_file='model.png')  
