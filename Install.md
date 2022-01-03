### Install

原文档链接: https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md

install的核心是保证cuda，pytorch，mmcv与mmdetection的版本互相一致。

1. 创新虚拟环境，准备pytorch

   `````sh
   conda create -n open-mmlab python=3.7 -y
   conda activate open-mmlab
   # 安装对应cuda版本的pytorch
   ## https://pytorch.org/get-started/previous-versions/
   pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
   `````

   

2. 根据环境安装mmcv

```sh
# 查看cuda,pytorch版本 
nvcc -V 
python -c "import torch; print(torch.__version__)"

# pip安装mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
# eg pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

## 本地编译安装mmcv
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
```

3. 安装mmdetection

   `````sh
   git clone https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   pip install -r requirements/build.txt
   pip install -v -e .  # or "python setup.py develop"
   `````

   

4. 测试安装是否成功

   ```sh
   # for test_rnv
   wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
   ```

   

   `````python
   from mmdet.apis import init_detector, inference_detector, show_result_pyplot
   config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
   # download the checkpoint from model zoo and put it in `checkpoints/`
   # url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
   checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
   device = 'cuda:0'
   # init a detector
   model = init_detector(config_file, checkpoint_file, device=device)
   # inference the demo image
   img = 'demo/demo.jpg'
   result = inference_detector(model, img)
   show_result_pyplot(model, img, result)
   `````

   