from face_register import FaceFeature

import os
import numpy as np
import pickle
import cv2
import shutil
from tqdm.auto import tqdm
from PIL import Image, ImageDraw, ImageFont


class VideoDetector():
    def __init__(self, faces_path, det_path, rec_path):
        self.face_feature = FaceFeature(det_path, rec_path)
        self.embeddings, self.idxes, self.name2idx = self.register_face(faces_path)
        self.idx2name = {idx:name for name, idx in self.name2idx.items()}

    def register_face(self, faces_path):
        print("Register Faces")
        # faces_path = "/zhangxin/103.视频人物检测/person_detection_with_pytorch/video_person_detection/faces"
        embeddings, idxes, name2idx = self.face_feature.get_embeddings_with_idx(faces_path)
        return embeddings, idxes, name2idx

    def recoginze_with_embedding(self, face_embedding, thresh):
        scores = []
        for embedding in self.embeddings:
            scores.append(self.face_feature.recognizer.cosine_distance(face_embedding, embedding))
        ok_idxs = []
        ok_scores = []
        for idx, score in enumerate(scores):
            if score >= thresh:
                ok_idxs.append(self.idxes[idx])
                ok_scores.append(score)
        return ok_idxs, ok_scores

    def cv2ImgAddText(self, img, text, left, top, textColor=(0, 255, 0), textSize=20):
        if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
            img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        fontText = ImageFont.truetype("../font/SimHei.ttf", textSize, encoding="utf-8")
        draw.text((left, top), text, textColor, font=fontText)
        return np.asarray(img)

    def detec_video(self, video_path, result_path, thresh=0.6, frame_interval=20, start_position=0):
        print("Detect Video") 
        # 清除results文件夹内容
        if os.path.exists(result_path):
            shutil.rmtree(result_path)
        os.mkdir(result_path)
        
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            print("无法读取视频!!!")
            return None

        capture.set(3, 640) # 设置视频的宽度
        capture.set(4, 480) # 设置视频的高度

        # 获取视频参数
        TOTAL_FRAME_NUMS = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for i, idx_frame in enumerate(tqdm(range(TOTAL_FRAME_NUMS))):
            ret, img = capture.read()
            # 如果没有读到
            if ret == -1:
                continue
            # 满足采样步骤的帧才处理
            if (i + 1)%frame_interval != 0:
                continue
            faces, boxes = self.face_feature.detect.get_faces(img)
            # 如果没有人脸,则进入下一帧
            if faces is None or boxes is None:
                continue

            assert len(faces) == len(boxes), "faces的长度必须和boxes的长度相同"
            for idx_face in range(len(faces)):
                face, box = faces[idx_face], boxes[idx_face]
            # for face, box in enumerate(faces, boxes):
                face_embedding = self.face_feature.recognizer.get_face_embedding(face)
                face_ids, face_scores = self.recoginze_with_embedding(face_embedding, thresh)

                for face_id, face_score in zip(face_ids, face_scores):
                    name = self.idx2name[face_id]
                    # name = str(face_id)
                    x, y, x_w, y_h = box
                    w = x_w - x
                    h = y_h - y
                    cv2.rectangle(img, (x,y), (x_w,y_h), (0,255,0), 2)
                    # cv2.putText(img, name, (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                    img = self.cv2ImgAddText(img, name, x+5, y-20, textColor=(255,255,255), textSize=20)
                    cv2.putText(img, "%.2f%%"%(face_score*100), (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                    cv2.imwrite(os.path.join(result_path, str(i)+".jpg"), img)


if __name__ == "__main__":
    det_path = "../onnx_models/version-RFB-640.onnx"   
    rec_path = "../onnx_models/face_reco.onnx"      
    faces_path = "../faces"
    # video_path = "../videos/郑爽.mp4"
    video_path = "../videos/范冰冰3.mp4"
    
    video_detector = VideoDetector(faces_path, det_path, rec_path)
    video_detector.detec_video(video_path, "../results")
