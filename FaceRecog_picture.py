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


def pre_calculate_dis(face_list):
    '''
    input: face_list 人脸库路径
    :return: 包含了预先计算好的人脸嵌入向量, 格式 numpy.array
    '''
    images = []
    for img_path in face_list:
        img = cv2.imread(img_path)
        img = img[:, :, :: -1]  # BGR转换为RGB
        images.append(img)
    return images

def main(args):

    start = time.time()
    img_ = cv2.imread(args.image_file)

    mt = mtcnn(160)
    img_ = img_[:, :, :: -1]  # BGR转换为RGB
    image1, rectangles_ = mt.detectFace(img_)
    print(rectangles_)
    image1 = prewhiten(image1[0])
    print(image1)

    with tf.Graph().as_default():

        with tf.Session() as sess:
            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")


            cap = cv2.VideoCapture(0)
            print(image1.shape)
            image1 = np.expand_dims(image1, axis=0)
            print(image1.shape)
            cnt = 0
            while (1):
                start1 = int(round(time.time()*1000))  # ms
                ret, img = cap.read()
                #cv2.imshow("333", img)
                #cv2.waitKey(0)
                img = img[:, :, :: -1]  # BGR转换为RGB
                image2, rectangles = mt.detectFace(img)
                # 再折腾回来
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if len(image1)==0 or len(image2)==0:
                    print("image1 or image2 is empty!")
                    cv2.imshow("face", img)
                    cv2.waitKey(10)
                    continue

                for i in range(len(image2)):
                    image2_ = copy.deepcopy(prewhiten(image2[i]))
                    _image2 = np.expand_dims(image2_, axis=0)
                    images = np.concatenate((image1, _image2), axis=0)
                    # Run forward pass to calculate embeddings
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[0, :], emb[1, :]))))
                    sim = calculSimilar(emb[0, :], emb[1, :])
                    print("相似度: ", sim)
                    print("Distance: ", dist)
                    cv2.rectangle(img, (int(rectangles[i][0]), int(rectangles[i][1])), (int(rectangles[i][2]), int(rectangles[i][3])), (255, 0, 0), 1)
                    cv2.putText(img, 'Similar: ' + str(sim)[:5], (int(rectangles[i][0]) + 5, int(rectangles[i][1])+ 5), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                cv2.imshow("face", img)
                if cv2.waitKey(10) and 0xFF == ord('q'):
                    break
                start2 = int(round(time.time()*1000))
                print("cost time(ms): ", (start2-start1))
                print("fps: ", 1000.0/(start2 - start1))
            cap.release()
            cv2.destroyAllWindows()



def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def load_and_align_data(img, image_size, margin, gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            print("can't detect face, remove ")
            return None
        det = np.squeeze(bounding_boxes[0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)

    return prewhitened


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
    parser.add_argument('image_file', type=str, help='Images to compare')
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
