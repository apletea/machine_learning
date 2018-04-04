from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Activation
from keras.models import Model, load_model
from keras.applications.vgg19 import preprocess_input,decode_predictions
import numpy as np

def get_VGG19():
    model = VGG19(weights='imagenet',include_top=False,input_shape=(75, 75, 3))

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='selu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64)(x)
    x = Activation('selu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    return Model(inputs=model.input, outputs=predictions)


def pop(self):
    '''Removes a layer instance on top of the layer stack.'''
    if not self.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    else:
        self.layers.pop()
        if not self.layers:
            self.outputs = []
            self.inbound_nodes = []
            self.outbound_nodes = []
        else:
            self.layers[-1].outbound_nodes = []
            self.outputs = [self.layers[-1].output]
        self.built = False


def pop_last_3_layers(model):
    pop(model)
    pop(model)
    pop(model)
    return model

def export_weight(model,filepath):
    model.save_weights(filepath)

def import_weight(model, filepath):
    model.weights(filepath, by_name=False)



model = get_VGG19()
img_path = '/home/davinci/Downloads/samurai-16.jpg'
img = image.load_img(img_path,target_size=(75,75))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
print 2
print x
x = preprocess_input(x)
print 3
print x

features = model.predict(x)
print 4
print features
