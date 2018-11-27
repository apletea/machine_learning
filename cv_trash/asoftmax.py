def Loss_ASoftmax(x, y, l, num_cls, m = 2, name = 'asoftmax'):
    '''
    x: B x D - data
    y: B x 1 - label
    l: 1 - lambda 
    '''
    xs = x.get_shape()
    w = tf.get_variable("asoftmax/W", [xs[1], num_cls], dtype=tf.float32, 
            initializer=tf.contrib.layers.xavier_initializer())

    eps = 1e-8

    xw = tf.matmul(x,w) 

    if m == 0:
        return xw, tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=xw))

    w_norm = tf.norm(w, axis = 0) + eps
    logits = xw/w_norm

    if y is None: 
        return logits, None

    ordinal = tf.constant(list(range(0, xs[0])), tf.int64)
    ordinal_y = tf.stack([ordinal, y], axis = 1)

    x_norm = tf.norm(x, axis = 1) + eps

    sel_logits = tf.gather_nd(logits, ordinal_y)

    cos_th = tf.div(sel_logits, x_norm)

    if m == 1:

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

    else:

        if m == 2:

            cos_sign = tf.sign(cos_th)
            res = 2*tf.multiply(tf.sign(cos_th), tf.square(cos_th)) - 1

        elif m == 4:

            cos_th2 = tf.square(cos_th)
            cos_th4 = tf.pow(cos_th, 4)
            sign0 = tf.sign(cos_th)
            sign3 = tf.multiply(tf.sign(2*cos_th2 - 1), sign0)
            sign4 = 2*sign0 + sign3 - 3
            res = sign3*(8*cos_th4 - 8*cos_th2 + 1) + sign4
        else:
            raise ValueError('unsupported value of m')

        scaled_logits = tf.multiply(res, x_norm)

        f = 1.0/(1.0+l)
        ff = 1.0 - f
        comb_logits_diff = tf.add(logits, tf.scatter_nd(ordinal_y, tf.subtract(scaled_logits, sel_logits), logits.get_shape())) 
        updated_logits = ff*logits + f*comb_logits_diff

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=updated_logits))

return logits, loss


class ASoftmax(Dense):
    def __init__(self, units, m,
                 batch_size = batch_size,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Dense, self).__init__(**kwargs)
        self.units = units
        self.m = m
        self.batch_size = batch_size
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True


    def call(self, inputs):
        inputs.set_shape([self.batch_size, inputs.shape[-1]])
        inputs_norm = K.sqrt(K.sum(K.square(inputs), axis=-1, keepdims=True))
        kernel_norm = tf.nn.l2_normalize(self.kernel, dim=(0, 1))    
        inner_product = K.dot(inputs, kernel_norm)         
        dis_cosin = inner_product / inputs_norm    
        m_cosin = multipul_cos(dis_cosin, self.m)   
        sum_y = K.sum(K.exp(inputs_norm * dis_cosin), axis=-1, keepdims=True)  
        k = get_k(dis_cosin, self.units, self.batch_size) 
        psi = np.power(-1, k) * m_cosin - 2 * k
        e_x = K.exp(inputs_norm * dis_cosin)
        e_y = K.exp(inputs_norm * psi)
        sum_x = K.sum(e_x, axis=-1, keepdims=True)
        temp = e_y - e_x
        temp = temp + sum_x
        output = e_y / temp
        return output

def multipul_cos(x, m):
    if m == 2:
        x = 2 * K.pow(x, 2) - 1
    elif m == 3:
        x = 4 * K.pow(x, 3) - 3 * x
    elif m == 4:
        x = 8 * K.pow(x, 4) - 8 * K.pow(x, 2) + 1
    else:
        raise ValueError("To high m")
    return x

def get_k(m_cosin, out_num, batch_num):
    theta_yi = tf.acos(m_cosin)  #[0,pi]
    theta_yi = tf.reshape(theta_yi, [-1])
    pi = K.constant(3.1415926)

    def cond(p1, p2, k_temp, theta):
        return K.greater_equal(theta, p2)

    def body(p1, p2, k_temp, theta):
        k_temp += 1
        p1 = k_temp * pi / out_num
        p2 = (k_temp + 1) * pi / out_num
        return p1, p2, k_temp, theta


    k_list = []
    for i in range(batch_num * out_num):
        k_temp = K.constant(0)
        p1 = k_temp * pi / out_num
        p2 = (k_temp + 1) * pi / out_num
        _, _, k_temp, _ = tf.while_loop(cond, body, [p1, p2, k_temp, theta_yi[i]])
        k_list.append(k_temp)
    k = K.stack(k_list)
    k = tf.squeeze(K.reshape(k, [batch_num, out_num]))
    return k

def asoftmax_loss(y_true, y_pred):
    d1 = K.sum(tf.multiply(y_true, y_pred), axis=-1)
    p = -K.log(d1)
    loss = K.mean(p)
    K.print_tensor(loss)
    return p
