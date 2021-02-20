- 新建videos文件夹,将待检测视频放入该文件夹下
- 新建results文件夹,每个视频检测结果对应一个视频名字命名的文件夹
- 新建fonts文件夹,将中文字体文件放入其中(SimHei.ttf)

本repo使用人脸检测的onnx模型及onnxruntime来运行人脸检测模型

主要的代码在codes文件夹下,运行如下命令

```python
python main.py
```

将使用faces中的人脸模板库,对videos中的视频文件进行检测,并将检测结果以图片的形式存在results下的对应文件夹中.
