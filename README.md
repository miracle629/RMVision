# RMVision
为Robomaster视觉任务提供一些想法和实现

目标检测与追踪任务采取tracking by detection策略，检测器包括Cascade、SVM、YOLO，追踪器包括Template、KCF、Staple。

1.cascade_TM_judge_color_beta(改自cascade_TM_judge_color)
cascade算法在灰度图上使用，robomaster比赛装甲板有红蓝两种颜色，需要judge_color，可选择在bgr或hsv空间进行judge。可视化部分包括save_board和 save_location，用于分析追踪失效和漏检帧。

2.Cascade_Template
追踪靠颜色直方图的模板匹配，搜索区域限制在以上一秒bounding box中心为中心的四倍面积区域内，对追踪准确性贡献很大。在追踪不丢失的情况下，每隔一定帧重新进行一次Cascade检测，防止过长时间丢失目标。Cascade速度快，模板匹配速度快，所以此算法在这一系列算法内速度最快，在测试视频中达到1300fps。对于光线强度敏感，模板匹配不够鲁棒，是其弱点。

3.Cascade_KCF
将追踪部分改为KCF，OpenCV中的KCF算法要求init和update的搜索帧大小相同，一般使用时直接对摄像头全画面搜索，因此速度较慢，此算法动态地在全画面内截取前一帧bounding box为中心，边长为150pixel的区域作为搜索区域，大幅度提高帧率，因搜索限制又提高了准确性。KCF容易在某一帧失效后之后的几帧连续失效，但update正常，陷入死循环。

3.1 将程序里大部分double类的计算改为int类计算，帧率1.8x。

4.Cascade_KCF-Template
穷人版的Staple跟踪，要求两种跟踪算法有一部分重叠才视为跟踪成功，一定程度减少了KCF跟踪失败陷入死循环的情况。

4.1 将程序里大部分double类的计算改为int类计算，帧率2.8x。

5.Cascade_Staple
Staple跟踪实际是HOG特征的KCF在线学习和颜色特征的在线学习互补的跟踪算法，算法鲁棒速度也较快。但限制是目前只能在WINDOWS下使用，作者fork了一个Ubuntu下的Staple项目，使用fftw3和eigen进行ubuntu下的傅里叶变换和矩阵计算，但在tx2上编译有问题还未解决，此代码在windows下编译。
