from video_detector import VideoDetector
import os
import configparser

class Main():
    def __init__(self, config_path="./config.ini"):
        cf = configparser.ConfigParser()
        cf.read(config_path)
        self.faces_path = "../faces"
        self.video_path = "../videos"
        self.results_path = "../results"
        self.thresh = float(cf.get("video-args", "thresh"))
        self.frame_interval = int(cf.get("video-args", "frame_interval"))
        self.video_detector = VideoDetector(self.faces_path, "../onnx_models/version-RFB-640.onnx", "../onnx_models/face_reco.onnx")

    def get_result_path(self, video_path):
        _, video_name = os.path.split(video_path)
        return os.path.join(self.results_path, video_name.split(".")[0])

    def detect_videos(self):
        video_name_list = os.listdir(self.video_path)
        for video_name in video_name_list:
            print("正在检测视频[%s]"%video_name)
            video_path = os.path.join(self.video_path, video_name)
            self.video_detector.detec_video(video_path, self.get_result_path(video_path),
                                            thresh=self.thresh, frame_interval=self.frame_interval)

if __name__ == "__main__":
    main = Main()
    main.detect_videos()