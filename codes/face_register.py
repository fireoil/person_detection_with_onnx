from face_detect import FaceDetector
from face_recognize import FaceRecognizer
import os
import numpy as np
import pickle
import cv2


class FaceFeature():
    def __init__(self, det_path, rec_path):
        self.detect = FaceDetector(det_path)
        self.recognizer = FaceRecognizer(rec_path) 

    def get_vocab(self, faces_dir):
        idx2name = os.listdir(faces_dir)
        name2idx = {name:idx for idx, name in enumerate(idx2name)}
        return name2idx

    def get_embeddings_with_idx(self, faces_dir):
        name2idx = self.get_vocab(faces_dir)
        names = os.listdir(faces_dir)
        embeddings = []
        idxes = []

        for name in names:
            # 取平均得到每个idx对应的embedding取均值
            # 这么做是为了按照顺序获取,好吧,是我的执念
            idxes.append(name2idx[name])
            avg_embedding = []

            image_paths = os.listdir(os.path.join(faces_dir, name))
            for image_path in image_paths:
                img_abspath = os.path.join(faces_dir, name, image_path)
                register_face = self.detect.detect_face(img_abspath)
                register_embedding = self.recognizer.get_face_embedding(register_face)
                avg_embedding.append(register_embedding)

            embeddings.append(np.array(avg_embedding).mean(axis=0))
        return embeddings, idxes, name2idx


if __name__ == "__main__":
    det_path = "../onnx_models/version-RFB-640.onnx"
    rec_path = "../onnx_models/face_reco.onnx"
    faces_path = "../faces"
    face_feature = FaceFeature(det_path, rec_path)
    embeddings, idxes, name2idx = face_feature.get_embeddings_with_idx(faces_path)
    print(idxes)
