import numpy as np
import onnx
import onnxruntime as ort
import cv2

class FaceRecognizer():
    def __init__(self, onnx_path):
        self.sess = ort.InferenceSession(onnx_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def get_onnx_input_tensor(self, img_np):
        img_np = np.array(img_np)
        image = cv2.resize(img_np, (112, 112))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        return image
    
    def get_face_embedding(self, img_np):
        img_np = self.get_onnx_input_tensor(img_np)
        ort_data = {self.input_name: img_np}
        embedding = self.sess.run([self.output_name], ort_data)[0].squeeze()
        return embedding

    def cosine_distance(self, a, b):
        if a.shape != b.shape:
            raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
        if a.ndim==1:
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
        elif a.ndim==2:
            a_norm = np.linalg.norm(a, axis=1, keepdims=True)
            b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        else:
            raise RuntimeError("array dimensions {} not right".format(a.ndim))
        similiarity = np.dot(a, b.T)/(a_norm * b_norm) 
        return similiarity


if __name__ == "__main__":
    face_recognizer = FaceRecognizer("../onnx_models/face_reco.onnx")
    test1_embedding = face_recognizer.get_face_embedding(cv2.cvtColor(cv2.imread("../results/test1.jpg"), cv2.COLOR_BGR2RGB))
    test2_embedding = face_recognizer.get_face_embedding(cv2.cvtColor(cv2.imread("../results/test2.jpg"), cv2.COLOR_BGR2RGB))
    test3_embedding = face_recognizer.get_face_embedding(cv2.cvtColor(cv2.imread("../results/test3.jpg"), cv2.COLOR_BGR2RGB))

    print("1和2, %f"%face_recognizer.cosine_distance(test1_embedding, test2_embedding))
    print("1和3, %f"%face_recognizer.cosine_distance(test1_embedding, test3_embedding))
    print("2和3, %f"%face_recognizer.cosine_distance(test2_embedding, test3_embedding))
