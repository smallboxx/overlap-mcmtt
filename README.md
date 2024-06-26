# overlap-mcmtt
基于区域重叠度多视角跟踪

# WILDTRACK数据集BEV方向重叠掩码
| 图像1 | 图像2 | 图像3 |
|-------|-------|-------|
| ![Figure_1](https://github.com/smallboxx/overlap-mcmtt/assets/127008146/b5641fed-08cf-4726-a547-6b17136c7265) | ![Figure_1mask_wildtrack](https://github.com/smallboxx/overlap-mcmtt/assets/127008146/878ea8b3-64f9-4af0-89fb-c96ac58ed742) | ![Figure_2mask_wildtrack](https://github.com/smallboxx/overlap-mcmtt/assets/127008146/bde47325-049b-4127-bf11-19b479cee4a2) |
| ![Figure_3mask_wildtrack](https://github.com/smallboxx/overlap-mcmtt/assets/127008146/94fac25e-9088-4b03-8261-40e08cad04d1) | ![Figure_2](https://github.com/smallboxx/overlap-mcmtt/assets/127008146/d295de11-59ab-468a-bc22-14d63f6e7879) |       |

# 前帧运动估计
| 图像1 | 图像2 | 图像3 |
|-------|-------|-------|
|![towards_vis1](https://github.com/smallboxx/overlap-mcmtt/assets/127008146/2cb25ed8-75b8-498f-9a9f-316a4dafeeb6)|![towards_vis2](https://github.com/smallboxx/overlap-mcmtt/assets/127008146/1e8532f9-db2d-4792-8337-96d68bee2c0d)|![towaards_vis3](https://github.com/smallboxx/overlap-mcmtt/assets/127008146/3ac61aa9-2c8b-4c63-9fb4-857e370f569e)



 
# EarlyBird Baseline
## 4090 24GB cuda118 torch2.1.0
## Baseline
|moda|modp|mota|motp|idf1|idp|fp|size
|----|----|----|----|----|----|----|----|
|89.71|82.33|88.34|11.04|93.04|91.99|65|99.38MB|

# 引入重叠度解码+时序信息融合+前帧运动估计
|moda|modp|mota|motp|idf1|idp|fp|size
|----|----|----|----|----|----|----|----|
|90.86|81.20|90.34|12.55|94.72|95.22|39|67.31MB|

# 可视化，多视角行人追踪，姿态估计，面部识别，年龄性别估计

|图1|图2|图3|
|--|--|--|
|![image](https://github.com/smallboxx/overlap-mcmtt/assets/127008146/9ecae98a-a4bc-4eca-a290-82d16344e3df)|![image](https://github.com/smallboxx/overlap-mcmtt/assets/127008146/1e48f463-7f92-4ae6-a64c-8edc26fd6864)|![image](https://github.com/smallboxx/overlap-mcmtt/assets/127008146/beeb5dad-01e9-440f-84ea-fee9b543c14d)

# Flask数据可视化平台
|图1|图2|
|--|--|
|![image](https://github.com/smallboxx/overlap-mcmtt/assets/127008146/ba06b306-de1e-48aa-b9b1-2c6990d90013)|![image](https://github.com/smallboxx/overlap-mcmtt/assets/127008146/f2c8a9d4-925c-4cb7-bfe2-92a589eb119c)|




