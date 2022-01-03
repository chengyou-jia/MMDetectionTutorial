### Tool （有用的脚本工具）

Reference：https://github.com/open-mmlab/mmdetection/blob/master/docs/useful_tools.md

在tools文件夹下，提供了训练与测试的script，同时也提供了很多的其他的有用工具。

#### 1. train

- 单gpu训练

````sh
CONFIG=_  # 配置文件路径
WORKDIR=_ # 结果保存目录
python ./tools/train.py $CONFIG  --work-dir $WORKDIR
# 其他的参数可以详见train.py文件或在config文件内修改
````



- 多gpu训练

```sh
GPU_NUM=_ # 使用gpu数量
CONFIG=_  # 配置文件路径
WORKDIR=_ # 结果保存目录
CUDA_VISIBLE_DEVICES=_ bash ./tools/dist_train.sh $CONFIG $GPU_NUM --work-dir $WORKDIR
# 其他的参数可以详见train.py文件或在config文件内修改
```

#### 2. test

- 单gpu测试

  ```sh
  CONFIG=_
  CHECKPOINT=_
  python ./tools/test.py $CONFIG $CHECKPOINT --out $OUTPUTFILE --eval bbox
  ```

  

- 多gpu测试

  ```sh
  CONFIG=_
  CHECKPOINT=_
  GPU_NUM=_
  CUDA_VISIBLE_DEVICES=_ ./tools/dist_test.sh $CONFIG $CHECKPOINT $GPU_NUM --out $OUTPUTFILE --eval segm
  ```

#### 3. analysis_tools

- 日志分析

```sh
# 安装 pip install seaborn
LOGFILE=_ # log文件 log.json
OUTFILE=_ # 图片输出地址
KEYS=_   # 打印的键值
TITLE=_  # 输出图片title
python tools/analysis_tools/analyze_logs.py plot_curve $LOGFILE [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUTFILE}]
# eg
# python tools/analysis_tools/analyze_logs.py plot_curve logo_train/20210723_033839.log.json --keys bbox_mAP --legend bbox_mAP
```

- 计算平均训练时长

  ```sh
  python tools/analysis_tools/analyze_logs.py cal_train_time $LOGFILE
  # 结果示例
  # -----Analyze train time of logo_train/20210723_033839.log.json-----
  # slowest epoch 7, average time is 0.3195
  # fastest epoch 12, average time is 0.3126
  # time std over epochs is 0.0018
  # average iter time: 0.3156 s/iter
  ```

- Test预测结果展示

  ```sh
  CONFIG=_ #配置文件
  PREDICTION_PATH_=_ #test预测的结果文件(.pkl)
  SHOW_DIR_=_ # 保存结果的目录
  # --show 是否直接展示结果 选择false
  WAIT_TIME=_ #直接展示结果的等待时长
  TOPK=_ #展示前几个结果
  SHOW_SCORE_THR=_ #展示结果的阈值
  CFG_OPTIONS=_ #配置文件的选项，默认为config文件
  python tools/analysis_tools/analyze_results.py \
        ${CONFIG} \
        ${PREDICTION_PATH} \
        ${SHOW_DIR} \
        [--show] \
        [--wait-time ${WAIT_TIME}] \
        [--topk ${TOPK}] \
        [--show-score-thr ${SHOW_SCORE_THR}] \
        [--cfg-options ${CFG_OPTIONS}]
  ```

- coco_error_analysis 结果分析，每个类上的分数展示

  ```sh
  # 获取json格式的结果文件
  # out: results.bbox.json and results.segm.json
  CONFIG=_
  CHECKPOINT=_
  RESULT_DIR=_
  ANN_FILE=_
  python tools/test.py \
         $CONFIG \
         $CHECKPOINT \
         --format-only \
         --options "jsonfile_prefix=./results"
  
  # 使用coco_error_analysis进行每个类的结果分析
  python tools/analysis_tools/coco_error_analysis.py \
         results.bbox.json \
  	   $RESULT_DIR    \
         --ann=$ANN_FILE \
  ```

  

- 模型复杂度分析

```sh
CONFIG_FILE=_
INPUT_SHAPE=_ # default : (1, 3, 1280, 800)
# FLOPs 与输入大小有关 parameters 与输入大小无关
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]

# 输出示例
# ==============================
# Input shape: (3, 1280, 800)
# Flops: 206.72 GFLOPs
# Params: 41.18 M
# ==============================

```

#### 4. 可视化

```sh
# https://github.com/Chien-Hung/DetVisGUI/tree/mmdetection
CONFIG_FILE=_ #  Config file of mmdetction.
RESULT_FILE=_ # pickle / json format.
STAGE=_ # train val test ,default is 'val'.
SAVE_DIRECTORY=_ # default is 'output'
python DetVisGUI.py ${CONFIG_FILE} [--det_file ${RESULT_FILE}] [--stage ${STAGE}] [--output ${SAVE_DIRECTORY}]
```



#### 5. 模型转换与模型部署

