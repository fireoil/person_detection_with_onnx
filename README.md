本repo使用人脸检测的onnx模型及onnxruntime来运行人脸检测模型

主要的代码在codes文件夹下,运行如下命令

```python
python main.py
```

将使用faces中的人脸模板库,对videos中的视频文件进行检测,并将检测结果以图片的形式存在results下的对应文件夹中.