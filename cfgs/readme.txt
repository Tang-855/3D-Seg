Pointnext:
--------------------cfgs--------------------
存放modelnet40ply2048、s3dis、scannet、shapenetpart数据集的.yaml文件；
存放；modelnet40、s3dis_pix4point、scanobjectn、scanobjectnn_pix4pointn、shapenetpart_pix4point数据集的不同模型的.yaml文件。

***** 将自己数据集的.yaml配置文件放在cfg\shapenetpart文件夹下，换模型时只需要更改该路径。跑wheat数据集时使用mydatacfgs.yaml配置文件 ***** 

--------------------data--------------------
存放数据集的文件夹，可以把自己的数据集按照ShapeNetPart数据集的格式放在该文件夹下，添加数据集时参考其中的readme文件。
***** 配置自己的数据集，暂时先不修改名字，后期可将useddatasetxyz、20230515、synsetoffset2category.txt都换成自己的名字 *****

--------------------datautils--------------------
forafterpointDataLoader.py测试时，用于加载数据集的
存放自监督的损失函数以及其他函数的文件夹

--------------------docs--------------------
examples：modelnet40、s3dis、scannet、scanobjectnn、shapenetpart的数据集格式
projects：数据可视化、pointnetxt网络结构等图片

--------------------examples--------------------
classification：分类模型的train.py、pretrain.py文件
segmentation：语义分割模型的main.py训练、test_s3dis_6fold.py测试、vis_results.py可视化结果的文件
shapenetpart：
         --2wheatpredict：存放部件分割测试结果的文件夹，在testformetric中自定义
         --74wheat：存放部件分割测试结果的文件夹，在testformetric中自定义
         --log：存放训练日志，部件分割的存放在shapenetpart中，
         --multiviewparam：自监督学习中的数据增强过程，经过转换的多视图存放的文件夹，生成多视图的代码需要自己添加
         --wheatpredicts   小麦预测结果
         --main.py：部件分割的运行代码：（加载预训练模型、冻结网络层、修改配置文件）
         --self_super_main.py：自监督的运行代码
         --visualwheat.py：小麦结果可视化
         --calparamandflosp.py    算网络参数量和flosp
         --calthrouput.py    计算模型吞吐量
         --instanceleafsegment.py
         --makedataset.py   制作shapenet数据格式
         --pipelinework.py  删
         --Point_cloud_postprocessing.py  删
         --regiongrowing.py  区域生长，用于实例分割instanceleafsegment.py中引用
         --testformetric.py  测试miou，修改路径、batch、config
         --use_all_leaf_area.py    删
         --visualwheat   可视化，修改路径、batch、config
--------------------openpoints--------------------
cpp：底层
         --chamfer_dist
         --emd
         --pointnet2_batch
         --pointops
         --subsampling：下采样
dataset：
         --atom3d
         --graph_dataset
         --matterport3d
         --modelnet：modelnet数据集的加载和重新采样
         --molhiv
         --molpcba
         --parsers
         --pcqm4m
         --pcqm4mv2
         --s3dis：s3dis数据集
         --scannetv2：ScanNet数据集
         --scanobjectnn：ScanObjectNNHardest数据集
         --semantic_kitti：Semantic_Kitti数据集
         --shapenet：ShapeNet数据集
         --shapenetpart：
             pointmlpshapenetpart.py加载PointMlp时的数据集，运行自监督时就将pointmlpshapenetpart.py重命名为shapenetpart.py；
             self_supervise_shapenetpart.py加载自监督学习时的数据集，运行自监督时就将self_supervise_shapenetpart.py重命名为shapenetpart.py；
             parseg_shapenetpart.py加载部件分割时的数据集，运行部件分割时就将used_fine_tuning_shapenetpart.py重命名为shapenetpart.py。
loss：存放的损失函数，如cross_entropy.py交叉熵损失
models：
         --backbone：存放dgcnn、pointnext等模型
         --classification
         --layers：存放卷积、注意力、图卷积、上采样、下采样、knn、权重初始化等
         --reconstruction：重建
         *--segmentation：base_seg.自监督修改。     vit_seg.py（删）
optim：存放各类优化器，如sgdp.py优化器SGD、adamp.py优化器Adam等
scheduler：定义各类学习率衰减策略，如cosine_lr.py余弦衰减、tanh_lr.py等
transforms：point_transform的数据加载方式
utils：

--------------------pytorch-OpCounter-master--------------------
OpCounter的pytorch版本

--------------------script--------------------
download_s3dis.sh：下载s3dis数据集的终端运行命令
main_classification.sh：分类的终端运行命令
main_partseg.sh：部件分割的终端运行命令
main_segmentation.sh：语义分割的终端运行命令
profile_flops.sh：
test_all_in_one.sh：

--------------------temp--------------------


--------------------thop--------------------