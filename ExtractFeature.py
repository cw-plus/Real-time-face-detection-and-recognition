import os
import cv2
import tensorflow as tf
import numpy as np
import facenet
import scipy.io as sio


def pre_calculate_dis(face_list):
    '''
    input: face_list 人脸库路径
    :return: 包含了预先计算好的人脸嵌入向量, 格式 numpy.array
    '''
    features = []
    with tf.Graph().as_default():

        with tf.Session() as sess:
            # Load the model
            facenet.load_model('20180408-102900/')

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            for i in range(len(face_list)):
                img = cv2.imread(face_list[i])
                img = img[:, :, :: -1]  # BGR转换为RGB
                img = cv2.resize(img, (160, 160))
                image1 = prewhiten(img)
                image2 = np.expand_dims(image1, axis=0)
                # Run forward pass to calculate embeddings
                feed_dict = {images_placeholder: image2, phase_train_placeholder: False}
                emb = sess.run(embeddings, feed_dict=feed_dict)
                np.squeeze(emb)
                features.append(emb)
                print(emb)
            sio.savemat('features.mat', {'features': features})

    return None

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


if __name__ == '__main__':
    '''
    每人 20 张图像放到自己名字的文件夹里
    '''
    img_list = []
    for root, dirs, files in os.walk("./pic/", topdown=False):
        for name in files:
            if name.endswith('.png') or name.endswith('.jpg'):
                print((os.path.join(root, name)))
                img_list.append(os.path.join(root, name))
    print(img_list)
    pre_calculate_dis(img_list)


