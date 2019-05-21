#coding:utf-8
import os
import sys
import copy
import time
import cv2
import tensorflow as tf
import numpy as np
import argparse
from scipy import misc

import facenet
import align.detect_face
from mtcnn import mtcnn

import scipy.io as scio


names = ["name1", "2", "3", "name4"]

pre_feature_dic = scio.loadmat('features.mat')
pre_feature = pre_feature_dic['features']

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


def calculSimilar(embedding1, embedding2):
    assert len(embedding1) == len(embedding2)

    em1 = np.mat(embedding1)
    em2 = np.mat(embedding2)
    num = float(em1 * em2.T)
    denom = np.linalg.norm(em1) * np.linalg.norm(em2)
    cos = num / denom
    return 0.5 + 0.5 * cos

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.7)
    return parser.parse_args(argv)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main(parse_arguments(sys.argv[1:]))

    start = time.time()
    mt = mtcnn(160)
    # 20 个照片一个人

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            cap = cv2.VideoCapture(0)
            while (1):
                start1 = int(round(time.time()*1000))  # ms
                ret, img = cap.read()
                img = img[:, :, :: -1]  # BGR转换为RGB
                image2, rectangles = mt.detectFace(img)
                # 再折腾回来
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if len(image2)==0:
                    print("Not detect person!")
                    cv2.imshow("face", img)
                    cv2.waitKey(5)
                    continue

                for i in range(len(image2)): # 来自摄像头
                  max_index = -1
                  max_val = 0.0
                  for j in range(len(pre_feature)): # 来自预先提取的
                      image2_ = copy.deepcopy(prewhiten(image2[i]))
                      images = np.expand_dims(image2_, axis=0)

                      # Run forward pass to calculate embeddings
                      feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                      emb = sess.run(embeddings, feed_dict=feed_dict)
                      # dist = np.sqrt(np.sum(np.square(np.subtract(pre_feature[j], emb))))
                      
                      sim = calculSimilar(pre_feature[j], emb)
                      if (sim > max_val):
                         max_val = sim
                         max_index = j
                    
                  cv2.rectangle(img, (int(rectangles[i][0]), int(rectangles[i][1])), (int(rectangles[i][2]), int(rectangles[i][3])), (255, 0, 0), 1)
                  cv2.putText(img, names[j//20] + str(sim)[:5], (int(rectangles[i][0]) + 5, int(rectangles[i][1])+ 5), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                  cv2.imshow("face", img)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                start2 = int(round(time.time()*1000))
                print("cost time(ms): ", (start2-start1))
                print("fps: ", 1000.0/(start2 - start1))
            cap.release()
            cv2.destroyAllWindows()
