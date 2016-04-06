# kaggle-101-FacialKeypointsDetection

## Facial Keypoints Detection

> Detect the location of keypoints on face images

## 脸部关键点检测


在本问题中，要求计算面部关键点的位置，即关键点在图片中的百分比坐标。
因此该问题的机理就是 [0, 1] 范围内的数值拟合，当然了，这也是一个多输出的拟合器。

给定图片与其相应的 30 个标签的百分比位置，标签信息如下：

>
  left_eye_center_x            
  left_eye_center_y            
  right_eye_center_x           
  right_eye_center_y           
  left_eye_inner_corner_x      
  left_eye_inner_corner_y      
  left_eye_outer_corner_x      
  left_eye_outer_corner_y      
  right_eye_inner_corner_x     
  right_eye_inner_corner_y     
  right_eye_outer_corner_x     
  right_eye_outer_corner_y     
  left_eyebrow_inner_end_x     
  left_eyebrow_inner_end_y     
  left_eyebrow_outer_end_x     
  left_eyebrow_outer_end_y     
  right_eyebrow_inner_end_x    
  right_eyebrow_inner_end_y    
  right_eyebrow_outer_end_x    
  right_eyebrow_outer_end_y    
  nose_tip_x                   
  nose_tip_y                   
  mouth_left_corner_x          
  mouth_left_corner_y          
  mouth_right_corner_x         
  mouth_right_corner_y         
  mouth_center_top_lip_x       
  mouth_center_top_lip_y       
  mouth_center_bottom_lip_x    
  mouth_center_bottom_lip_y

其中标签完整的图片有 2140 张，其中，图片的大小为 96*96 pixels。

求解步骤如下：

    Step 1. 选择拟合器 SVR/KernelRidge 以及对应的 kernel
    Step 2. 交叉验证实验选择超参数，超参数的选择通过枚举的方法
    Step 3. 选定超参数后，用所有训练集训练拟合器
    Step 4. 对测试集做预测，并输出结果
    
### 结果：

  First idea: Using 30 fitter to fit 30 labels, then I got 3.48060 RMSE
  Second idea: Using 1 fitter to fit 30 labels, then I got 3.43998 RMSE
  Third idea: Adding symmetrical training data, then resulting in abnormal result, such as position was greater then 96.
            So, I can see that the result of fitting is only cover [0,96](or [0,1])

### 30 个拟合器超参数调试的方法与结果如下：

    超参数选择 gamma
    
      for G in G_para:
          scores = list()
          for i in range(3):
              X1, X2, y1, y2 = train_test_split(train_X, train_y, test_size=0.3, random_state=42)
              clf = KernelRidge(kernel='rbf', gamma=G, alpha=1e-2)
              pred = clf.fit(X1, y1).predict(X2)
              sco = calbais(pred, y2)
              scores.append(sco)
          print('G:', G, 'Score:', scores)
    
    拟合器 
      KernelRidge(kernel='rbf', gamma=2e-4, alpha=1e-2)
      
    0.7:0.3 训练集划分拟合误差：
    
      [0] 0.7792    [10] 0.9744    [20] 1.0985
      [1] 0.6383    [11] 0.7451    [21] 1.2300
      [2] 0.7714    [12] 0.9513    [22] 1.2636
      [3] 0.6482    [13] 0.9299    [23] 1.1784
      [4] 0.7355    [14] 1.0870    [24] 1.2469
      [5] 0.6005    [15] 1.1898    [25] 1.2440
      [6] 0.9636    [16] 0.9012    [26] 0.9444
      [7] 0.7063    [17] 0.9462    [27] 1.3718
      [8] 0.7214    [18] 1.1349    [28] 0.9961
      [9] 0.6089    [19] 1.1669    [29] 1.5076

### pandas usage:

    数据统计：DataFrame.count()
    数据去缺失项：DataFrame.dropna()
    字符串分割：Series = Series.apply(lambda im: numpy.fromstring(im, sep=' '))

### 值得注意的地方：

    镜像图片，似乎对本问题采用 kernel ridge 拟合器 的求解没有帮助。
