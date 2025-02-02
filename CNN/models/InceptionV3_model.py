from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


def model_imagenet(img_width,
                   img_height,
                   num_classes,
                   x_all=None,
                   y_all=None,
                   optimizer=None):
	base_model = InceptionV3(
	    weights='imagenet',
	    include_top=False,
	    input_shape=(img_height, img_width, 3))

	x = base_model.output
	x = GlobalAveragePooling2D(name='avg_pool')(x)

	predictions = Dense(1, activation='sigmoid')(x) if num_classes == 2 \
     else Dense(num_classes, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	return model, base_model


def InceptionV3_imagenet():
	return {
	    "model": model_imagenet,
	    "name": "InceptionV3_imagenet",
	    "shape": (224, 224, 3),
		"pretrained": True
	}


def model_sinpesos(img_width,
                   img_height,
                   num_classes,
                   x_all=None,
                   y_all=None,
                   optimizer=None):
	base_model = InceptionV3(
	    weights=None, include_top=False, input_shape=(img_height, img_width, 3))

	x = base_model.output
	x = GlobalAveragePooling2D(name='avg_pool')(x)

	predictions = Dense(1, activation='sigmoid')(x) if num_classes == 2 \
     else Dense(num_classes, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	return model, base_model


def InceptionV3_sinpesos():
	return {"model": model_sinpesos, "name": "InceptionV3_sinpesos"}


def model_imagenet_transfer(img_width,
                            img_height,
                            num_classes,
                            x_all=None,
                            y_all=None,
                            optimizer=None):
	base_model = InceptionV3(
	    weights='imagenet',
	    include_top=False,
	    input_shape=(img_height, img_width, 3))

	x = base_model.output
	x = GlobalAveragePooling2D(name='avg_pool')(x)

	predictions = Dense(1, activation='sigmoid')(x) if num_classes == 2 \
     else Dense(num_classes, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	return model, base_model


def InceptionV3_transfer():
	return {
	    "model": model_imagenet_transfer,
	    "name": "InceptionV3_transfer",
	    "shape": (224, 224, 3),
		"transfer": True
	}
