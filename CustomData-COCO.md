### CustomData-COCO

------

本文档将以COCO格式json文件标注为例，介绍如果产生对应的标注文件以利用mmdetection进行训练。

Reference: https://zhuanlan.zhihu.com/p/29393415



### COCO标注格式

COCO标注格式主要由image，categories，annotations 3个字典字段表示。

#### 整体结构

```json
{
    # "info": info, 选填
    # "licenses": [license],选填
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}
```

#### 1. images

Images是包含多个image实例的数组，对于一个image类型的实例：

```json
{
	"file_name":"文件名或文件路径",
	"height":360,"width":640,
	"id":391895 # 文件id
},
```

#### 2. annotations

annotations字段是包含多个annotation实例的一个数组，annotation类型本身又包含了一系列的字段，如这个目标的category id和segmentation mask。

```json
annotation{
    "id": int,  # 标注id
    "image_id":  # 图片文件id,
    "category_id": # 类别id,
    "segmentation": RLE or [polygon], # iscrowd=0，polygons格式;iscrowd=1，RLE格式）
    "area": float, # 标注区域的面积（如果是矩形框，那就是高乘宽）
    "bbox": [x,y,width,height], # box标注 x,y,w,d
    "iscrowd": 0 or 1, # 单个的对象（iscrowd=0);一组对象(比如一群人）(iscrowd=1)
}
```

#### 3. categories

categories是一个包含多个category实例的数组，而category结构体描述如下：

```json
{
    "id": int, # 类别id,
    "name": str, # 类别名
    "supercategory": str, # 类别父类 如 vehicle（bicycle） 选填
}
```

### COCO标注获取实例

````python
# 初始化id 
image_id = 1
annotation_id = 1
# 初始化输出
coco_output = {
        "images":[],
        "categories":[],
        "annotations":[]
    }
categories = [{'id': 1, 'name': '...'},
              {'id': 2, 'name': '...'},
              {'id': 3, 'name': '...',},
              ...
             ]
coco_output['categories'] = categories
# 遍历文件
for i,file_path in enumerate(image_list):
    # 获取图片信息 file_name,height,width
    image_dict = {
        'file_name': file_name,
        "height":height,
        "width":width,
        "id":image_id # 文件id        
    }
    coco_output["images"].append(image_dict)
    ### 获取图片对应boxs信信息 box
    for box in boxs:
        ann_dict = {
            "id":annotation_id, 
            "image_id":image_id,
            "category_id": box.category_id, # 类别id
            "bbox": box.bbox,#[x,y,w,h]
            "area": box.bbox.area,
            "iscrowd": 0, #单人或者多人
        }
        coco_output["annotations"].append(ann_dict)
        annotation_id += 1
        
    image_id += 1 
# 保存
json.dump(coco_output, open(out_path, 'w'))
````

### 检测标注文件是否正确

可以使用pycocotools工具或mmdetection可视化工具来检验得到的标注文件是否正确

#### 1. pycocotools

如果标注文件中含有segmentation字段可以使用pycocotools进行可视化

Reference：https://blog.csdn.net/gzj2013/article/details/82385164

```python
from pycocotools.coco import COCO
anno_file = "标注文件路径"
coco_coco = COCO(anno_file)
catNms = ["class1", 'class2',...] # 获取类别
CatIds = coco_coco.getCatIds(catNms=catNms)
ImgIds = coco_coco.getImgIds(catIds=CatIds)
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
img_path = os.path.join(image_dir,img['file_name'])
I = Image.open(img_path)        
plt.axis('off')
annIds = coco_coco.getAnnIds(imgIds=img['id'], catIds=CatIds, iscrowd=None)
anns = coco_coco.loadAnns(annIds)
coco_coco.showAnns(anns)
```

输出结果如下：

![image-20210722203220449](C:\Users\Eric\AppData\Roaming\Typora\typora-user-images\image-20210722203220449.png)

#### 2. mmdetection：tools/misc/browse_dataset.py

使用mmdetetion中的工具browse_dataset来验证

修改对应配置文件：

````python
dataset_type = 'COCODataset'
classes = ('class1','class2','class3',...)
data = dict(
    train=dict(
        type='CocoDataset',
        ann_file='dataset/train.json',
        classes=classes,
        img_prefix='...',
        ),
    val=dict(
        type='CocoDataset',
        ann_file='dataset/train.json',
        classes=classes,
        img_prefix='...',
        ),
    test=dict(
        type='CocoDataset',
        ann_file='dataset/train.json',
        classes=classes,
        img_prefix='...',
        ) 
    )
````

执行工具：

```bash
python tools/misc/browse_dataset.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_QRcode.py
```

输出结果如下：

![image-20210722203637411](C:\Users\Eric\AppData\Roaming\Typora\typora-user-images\image-20210722203637411.png)

