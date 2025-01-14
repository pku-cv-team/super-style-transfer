# 北京大学计算机视觉2024期末大作业——风格迁移

这个仓库托管北京大学计算机视觉2024期末大作业——风格迁移的相关代码，支持若干种常见的风格迁移。

## 部署

将仓库部署到本地，需要先下载代码，执行

```bash
git clone https://github.com/pku-cv-team/super-style-transfer.git
```

进入项目目录，即可查看文件。

## 环境搭建

为了运行项目，你需要有一个 `Python` 解释器，最好安装有 `Anaconda` 或 `MiniConda` ，用以管理虚拟环境。以 `Anaconda` 或 `MiniConda` 为例，可以依次执行下面的命令这样创建项目需要的环境：

```bash
conda create -n style_transfer python=3.12.8 matplotlib pytest opencv pycocotools
conda activate style_transfer
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

> [!NOTE] 环境搭建有困难？
> 可以使用我们打包的 Docker 镜像，使用命令
>
> ```bash
> docker pull
> ```
>
> 拉取镜像。

## 具体使用

### Gatys 风格迁移

#### 示例

可以直接使用我们的示例配置文件[config_gatys.json](./experiments/config_gatys.json)，在项目根目录执行下面的命令

```bash
python style_transfer/train.py --config experiments/config_gatys.json
```

第一次使用这个项目你可能需要先执行

```bash
export PYTHONPATH=$PWD
```

#### 具体解释和修改方式

你也可以修改配置文件，实验你想要的风格迁移。进入我们的示例文件[config_gatys.json](./experiments/config_gatys.json)，你会看到这样的内容

```json
{
  "content_image": "data/raw/content/cambridge.jpg",
  "style_image": "data/raw/style/starry_night.jpg",
  "model_config": {
    "type": "gatys",
    "feature_extractor": {
      "type": "vgg19",
      "content_layers": [22],
      "style_layers": [1, 6, 11, 20, 29]
    },
    "content_layer_weights": [1.0],
    "style_layer_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
    "content_weight": 1e3,
    "style_weight": 3e8,
    "additional_loss": [
      {
        "type": "tv_loss",
        "tv_weight": 1e-2
      },
      {
        "type": "lap_loss",
        "lap_weight": 1e-2,
        "pool_size": 4
      }
    ],
    "init_strategy": "content"
  },
  "resize_stragety": {
    "type": "srcnn",
    "size": [300, 300],
    "model_path": "experiments/pretrained_models/srcnn_x3.pth",
    "scale": 3
  },
  "learning_rate": 0.2,
  "iterations": 500,
  "device": "cuda",
  "optimizer": "LBFGS",
  "output_image": "experiments/results/gatys/01.jpg"
}
```

你可以修改内容图像和风格图像的路径，相对路径和绝对路径都是允许的，不支持 `url` 。对于模型配置，你可以修改其类型，但是我们的项目当前只支持 `gatys` ，修改为其他的都会收到报错。风格图像可以支持多种风格的混合，比如修改 `style_image` 为

```json
"style_images": ["data/raw/style/starry_night.jpg", "data/raw/style/picasso.jpg"]
```

而 `style_weight` 对应修改为

```json
"style_weight": [3e10, 3e10]
```

可以修改特征提取器，当前支持的类型为 `vgg19` 和 `resnet50` ，其他的类型没有支持。你也可以修改对应的内容网络层和风格网络层，对于 `vgg19` ，你需要指定其序号，序号从 $0$ 开始；对于 `resnet50` ，你需要指定其网络层的名字，当前只支持 `conv1` ， `layer1` ， `layer2` ， `layer3` ， `layer4` ，具体的可以查询[feature_extractor](./style_transfer/models/feature_extractor.py)。你可以设置每一个层对应的权重，默认情况下是一样的。你可以额外增加损失，增加 `lap_loss` 即为 `Lapstyle` 的实现，或者可以增加 `tv_loss` ，并设置其权重。 `lap_loss` 支持对多种池化大小进行加权求和，，比如修改为

```json
{
  "type": "lap_loss",
  "lap_weight": 1e9,
  "pool_size": [2, 4],
  "pool_weight": [0.75, 0.25]
}
```

你也可以调整图像大小调整策略，目前支持 `trivial` ， `srcnn` 和 `pyramid` ，其样例配置均已给出。需要注意的是，使用 `srcnn` 之前需要下载对应的模型，进入 `scripts` 目录，执行命令

```bash
python auto_srcnn_download.py
```

### 前馈网络风格迁移

使用前馈网络进行风格迁移，可以使用我们与训练的模型，发布在[这里](https://github.com/pku-cv-team/super-style-transfer/releases/tag/v1.0.2)，下载到本地，使用下面的命令

```bash
python style_transfer/fast_stylize.py --input <输入图片> --output <输出图片> --model <模型路径>
```

或者你也可以自己训练模型，可以使用命令

```bash
python style_transfer/fast_train.py --config experiments/config_fast.json
```

你可以自由的修改[config_fast.json](./experiments/config_fast.json)的内容调整训练，或者使用我们整理的笔记本[fast_neural_style.ipynb](./fast_neural_style.ipynb)，我们同样提供了可以直接在[Colab](https://colab.research.google.com/)运行的笔记本[fast_neural_style_for_colab](./fast_neural_style_for_colab.ipynb)。如果你在本地训练，那么你需要下载对应的数据集，在项目根目录执行

```bash
./script/get_coco.sh
```

## 框架简单介绍

下面简单介绍一下我们的代码。

### 数据

所有数据以图片形式放在项目根目录的 `data` 目录下， `data` 目录下有三个子目录， `processed` 存放预处理后的数据（一般不用）， `raw` 存放原来的图片， `raw` 目录下有两个子目录 `content` 和 `style` 目录，分别存放内容图片和风格图片， `coco` 用来存放 `COCO` 数据集。

### 代码

所有实验相关的代码存放在 `style_transfer` 目录下，并有 `__init__.py` 使其成为一个包。

#### 数据处理

目录下 `data.py` 负责数据处理相关工作，框架仅仅封装了基本的操作，用户需要自行组合。 `dataset` 封装了 `COCO` 数据的读取操作。

#### 训练和评估

目录下 `train.py` 和 `evaluate.py` 分别负责训练和评估模型，训练部分的代码基本完成，但有些许需要改进，评估部分尚未完成。训练过程需要配置较多的参数，为了实验的方便，框架采用读取配置文件的方式，相关的配置文件应该存放在 `experiments` 目录下，如 `config_gatys.json` ，使用是需要指定命令行参数 `config` 为对应的配置文件路径，你可以在配置文件中指定训练的参数、内容图像和风格图像的路径和图像保存路径，基本的样例参考[config_gatys.json](./experiments/config_gatys.json)。

#### 支持函数

目录下 `utils` 子目录存放一些常用的支持函数，作为子模块。其中 `func_utils.py` 实现了一些常用的函数工具， `json_loader.py` 封装了 `Json` 文件加载的功能， `visualization.py` 封装了一些关于可视化的函数，以及其他的一些封装。

#### 模型定义

模型定义为一个子模块，在 `models` 目录下， `feature_extractor.py` 定义了特征提取器，包含一个抽象类 `FeatureExtractor` ，你可以继承这个类实现自己的特征提取器，作为示例，框架实现了一个基于 `CNN` 的特征提取器，主要用预训练的 `VGG-19` 完成。 `neural_style_transfer.py` 封装了神经风格迁移的抽象类 `NeuralStyleTransferModel` ，用户需要重载 `forward` 方法，作为示例，我们在 `gatys.py` 中实现了 `GatysStyleTransferModel` 类，算法主要参考了[这篇论文](https://dl.acm.org/doi/10.1145/3123266.3123425)。为了灵活的添加新的功能，我们采用装饰器模式，在 `neural_style_transfer_decorator.py` 中定义了神经风格迁移的装饰器，用户可以通过重载 `forward` 方法添加功能。作为示例，我们在 `lapstyle.py` 中实现了 `LapStyleTransferModel` 类，算法主要参考了[这篇论文](https://ieeexplore.ieee.org/document/7780634)。

### 测试

我们在 `tests` 目录下定义了测试模块，所有单元测试文件应该放在这个目录下，目前实现了 `test_data.py` ， `test_json.py` 和 `test_visualization.py` ，你可以根据需要自行定义其他测试。测试采用 [pytest](https://docs.pytest.org/en/stable/index.html) 框架，可以通过运行

```bash
pytest
```

运行所有的测试。

### 基于前馈网络的代码

基于前馈网络的代码实现主要依照[论文](https://arxiv.org/abs/1603.08155)，参考了 `Pytorch` [样例仓库](https://github.com/pytorch/examples/tree/main) 的[代码](https://github.com/pytorch/examples/tree/main/fast_neural_style)。

### 数据

训练数据使用 [COCO](https://cocodataset.org/) 数据集，使用2017年的数据，将下载到 [data/coco](data/coco/) 目录下，为了使用的方便，框架提供了下载脚本，Linux平台可以使用[get_coco.sh](./get_coco.sh)脚本，其他平台可以使用[get_coco.py](./get_coco.py)脚本。

> [!NOTE]
> 我们用的是 [COCO](https://cocodataset.org/) 数据集，这个数据集很大，可能需要下载很久，本地训练也可以只使用验证集

### 基本组成

框架在[transfer_net.py](./style_transfer/models/transfer_net.py)定义了生成网络，在[loss_net.py](./style_transfer/models/loss_net.py)定义了损失网络，在[dataset.py](./style_transfer/dataset.py)定义了 `COCO` 数据集，定义了训练模块[fast_train.py](./style_transfer/train.py)，应用模块[fast_stylize.py](./style_transfer/fast_stylize.py)。

训练的配置文件为[config_fast.json](./experiments/config_fast.json)。

### 训练

为了训练的方便，我们将所有的相关代码整理到了笔记本[fast_neural_style.ipynb](./fast_neural_style.ipynb)，可以在这里训练。
