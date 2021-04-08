import keras
from utils.load_cifar import load_data
from keras.preprocessing.image import ImageDataGenerator
from models import resnet, densenet, inception, vggnet
from compression import prune_weights, save_compressed_weights
from copy import deepcopy

import os


def save_history(history, result_dir, prefix):
    print(history.history.keys())
    loss = history.history['loss']
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, '{}_result.txt'.format(prefix)), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))


def schedule(epoch):
    if epoch < 20:
        return 0.1
    elif epoch < 40:
        return 0.01
    elif epoch < 60:
        return 0.001
    else:
        return 0.0001


def training(com_rate):
    fine_tune_epochs = 20
    batch_size = 128
    x_train, y_train, x_test, y_test, nb_classes = load_data('c10')
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    nb_classes = 10
    model = vggnet.vgg(nb_classes)
    model.summary()
    model.load_weights('./results10/vgg_c10_weights.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=5. / 32,
                                 height_shift_range=5. / 32)
    data_iter = datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True)

    # pre-train

    # prune weights
    # save masks for weight layers
    masks = {}
    layer_count = 0
    # not compress first convolution layer
    first_conv = True
    for layer in model.layers:
        weight = layer.get_weights()
        if len(weight) >= 2:
            if not first_conv:
                w = deepcopy(weight)  # 深拷贝
                tmp, mask = prune_weights(w[0], compress_rate=comp_rate)
                masks[layer_count] = mask
                w[0] = tmp
                layer.set_weights(w)
            else:
                first_conv = False
        layer_count += 1
    # evaluate model after pruning
    score = model.evaluate(x_test, y_test, verbose=0)
    print('val loss: {}'.format(score[0]))
    print('val acc: {}'.format(score[1]))
    # fine-tune
    for i in range(fine_tune_epochs):
        for _ in range(x_train.shape[0] // batch_size):
            X, Y = data_iter.next()
            # train on each batch
            model.train_on_batch(X, Y)
            # apply masks
            for layer_id in masks:
                w = model.layers[layer_id].get_weights()
                w[0] = w[0] * masks[layer_id]
                model.layers[layer_id].set_weights(w)
        score = model.evaluate(x_test, y_test, verbose=0)
        print('fine-tune: {}'.format(i))
        print('val loss: {}'.format(score[0]))
        print('val acc: {}'.format(score[1]))

    # save compressed weights
    compressed_name = './results10/compressed_{}_weights'.format(comp_rate)
    save_compressed_weights(model, compressed_name)


if __name__ == '__main__':
    #GPU config
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    #config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    for comp_rate in [0.75,0.78, 0.81, 0.84, 0.87, 0.93, 0.96]:
        training(comp_rate)
