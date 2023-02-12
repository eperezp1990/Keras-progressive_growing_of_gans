
def evaluate_fold(report_file, map_model, optimizer, train_x_f, train_y_f, test_x_f, test_y_f, train_extra_x_f, gpu=0):
    
    print('>> start', report_file)

    from Evaluation import Evaluate, SaveResults

    import time
    import pandas as pd
    import tensorflow as tf
    from keras import backend as k
    from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
    import numpy as np

    np.random.seed(7)

    epochs = 150
    epochs_pre = 30
    transfer = 'pretrained' in map_model

    train_x, train_y = np.load(train_x_f), np.load(train_y_f)

    unique, counts = np.unique(train_y, return_counts=True)
    less_label = unique[np.argmin(counts)]

    train_extra_x = np.load(train_extra_x_f)
    
    extra_y = np.asarray([less_label for _ in range(train_extra_x.shape[0])])

    train_x = np.vstack((train_x, train_extra_x))
    train_y = np.hstack((train_y, extra_y))

    test_x, test_y = np.load(test_x_f), np.load(test_y_f)

    with tf.device('/gpu:' + str(gpu)):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        try:
            k.set_session(sess)
        except:
            print('No session available')

        model, base_model = map_model['model'](224, 224, 2)

    # evaluation metrics
    evaluation = Evaluate(model, train_x, train_y, test_x, test_y)
    save_results = SaveResults(evaluation, report_file)

    # callbacks
    def callbaks_init():
        callbacks = [save_results]
        if optimizer == 'sgd':
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, mode='min', verbose=1))
        return callbacks

    def train_model(epochs):
        model.fit(train_x, train_y, epochs=epochs, batch_size=8, validation_data=(test_x, test_y), verbose=2,
                    callbacks= callbaks_init()
                 )

    countbase = len(base_model.layers)
    # freeze
    if transfer and epochs_pre > 0:
        for layer in model.layers[:countbase]:
            layer.trainable = False
        print('>> compile')
        with tf.device('/gpu:' + str(gpu)):
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            print('>> train transfer')
            train_model(epochs_pre)
    # unfreeze
    if transfer and epochs_pre > 0:
        for layer in model.layers[:countbase]:
            layer.trainable = True
    print('>> compile')
    with tf.device('/gpu:' + str(gpu)):
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        print('>> train normal')
        train_model(epochs)

