# Readme

##	<left><font size=3> 代码结构</left></font>

 >* `run.py` 主文件 程序起点 采用命令行运行  
 >* `comman` 调用训练测试以及PGD攻击  
 >* `net`  实现量化，正交和谱范正则，网络架构和PGD架构  
 >* `dataloader` 加载数据集  
 >* `dataset` 数据集  
 >* `pwla_relu` 动态阈值激活的实现  
 >* `util` squeeze的代码实现，保存模型，设置种子，以及可视化处理  

##	<left><font size=3> 运行方式</left></font>
>* `python run.py` 搭配各种参数使用

`--savepath` 训练过程中模型参数的存储位置  
`--imgpath` 训练过程中图片的存储位置  
`--quant`开启量化模式  
`--activate` 使用阈值激活函数  
`--regular`  使用正交正则  
`--spectral`  使用谱范正则  
`--PGD_train` 使用对抗训练  
`--iterations` 指定训练迭代次数  
`--eval`  开启测试模式，只在test集上做测评  
`--denorm` 不适用Normalize初始化图片  
`--squeeze` 指定压缩的方式 默认不压缩,0为颜色压缩,1为中值滤波，2为non_local-smoothing  

例如 `python run.py --quant --activate --savepath quant_model --imgpath quant ` 

表示将要以量化和动态阈值激活模式开启训练。
