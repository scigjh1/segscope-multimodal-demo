# 私有研究模型接口

公共 Demo 只发布交互与数据协议。未公开模型文件和权重放在仓库外，由操作者在本机配置：

```bash
export SEG_SCOPE_PREDICTOR_FILE=/path/to/private/predictor.py
python server.py --port 4175
```

PowerShell 使用 `$env:SEG_SCOPE_PREDICTOR_FILE='D:/private/predictor.py'`。本仓库不会读取 .env，环境变量须在启动进程前设置。接口用于可信操作者配置，不接受浏览器提交的模块路径。

适配器提供 `predict(image_bgr, auxiliary_gray, roi)`，输入分别为 uint8 BGR 图像、同尺寸灰度辅助图或 None、(x,y,width,height) 框。输入已经缩放到最长边 512。返回同尺寸的二维二值数组，值为 0/1 或 0/255；概率输出须在私有适配器内自行阈值化。

适配器模块在进程内缓存，可自行加载与缓存模型；根据私有模型的训练协议完成归一化、设备选择与窗口处理。公共接口不提供医院数据、训练权重或研究模型实现。

API 的 mode='private' 调用此接口；未配置时 UI 禁用此选项。不将模型路径返回给浏览器，禁止从 static 加载适配器。融合权重如不适用于研究模型，可设置为 0。

测试只使用几何掩膜验证接口契约，没有运行、验证或评估任何未公开医学研究模型。
