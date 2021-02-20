import cv2
import os
import os.path as osp
import onnxruntime as ort
import numpy as np
from dependencies.box_utils import predict

class FaceDetector():
    def __init__(self, onnx_path="../onnx_models/version-RFB-640.onnx"):
        self.face_detector = ort.InferenceSession(onnx_path)

    # scale current rectangle to box
    def scale(self, box):
        width = box[2] - box[0]
        height = box[3] - box[1]
        maximum = max(width, height)
        dx = int((maximum - width)/2)
        dy = int((maximum - height)/2)
        
        bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
        return bboxes

    # face detection method
    def faceDetector(self, orig_image, threshold = 0.7):
        image = cv2.resize(orig_image, (640, 480))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        input_name = self.face_detector.get_inputs()[0].name
        confidences, boxes = self.face_detector.run(None, {input_name: image})
        boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
        return boxes, labels, probs


    def crop_box(self, img, box):
        return img[box[1] : box[3],  box[0] :  box[2]]

    def crop_boxes(self, img, boxes):
        imgs = [None]*len(boxes)
        for idx, box in enumerate(boxes):
            imgs[idx] = self.crop_box(img, box)
        return imgs

    def detect_face(self, img_path, single=True):
        orig_image = cv2.imread(img_path)
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = self.faceDetector(orig_image)
        
        if len(boxes) < 1:
            # print("%s没有人脸"%img_path)
            return None
        else: 
            if single:
                return self.crop_box(orig_image, boxes[0])
            else:
                return self.crop_boxes(orig_image, boxes)

    def get_faces(self, img_np):
        boxes, labels, probs = self.faceDetector(img_np)
        
        if len(boxes) < 1:
            # print("没有人脸,请更换人脸")
            return None, None
        else: 
            return self.crop_boxes(img_np, boxes), boxes

if __name__ == "__main__":
    test_img_path = osp.join(cmd_root, "../faces/华晨宇/1.jpg")

    face_detector = FaceDetector()
    register_face = face_detector.detect_face(test_img_path)
    cv2.imwrite("../results/test3.jpg", cv2.cvtColor(register_face, cv2.COLOR_RGB2BGR))
