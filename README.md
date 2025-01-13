# 北京大学计算机视觉2024期末大作业——风格迁移

这个仓库托管北京大学计算机视觉2024期末大作业——风格迁移的相关代码，当前为基本框架，可以基于这个框架拓展其他功能，实验新颖的想法。框架基本完成了 `Gatys` 神经风格迁移和 `Lapstyle` 风格迁移，但存在许多的需要改进的地方，并可能有一些 bug 需要修补。

## 部署

将仓库部署到本地，需要先下载代码，执行

```bash
git clone https://github.com/pku-cv-team/super-style-transfer.git
```

进入项目目录，即可查看文件。

## 环境搭建

为了运行项目，你需要有一个 `Python` 解释器，最好安装有 `Anaconda` 或 `MiniConda` ，用以管理虚拟环境。以 `Anaconda` 或 `MiniConda` 为例，可以依次执行下面的命令这样创建项目需要的环境：

```bash
conda create -n style_transfer python=3.12.8 black pylint matplotlib pytest
conda activate style_transfer
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

## 项目运行

部署项目到本地并搭建好环境后，可以执行下面的命令运行项目中的 `Gatys` 风格迁移示例

```bash
python style_transfer/train.py --config experiments/config_gatys.json
```

就可以在 `experiments/results/gatys/` 下看到风格迁移的结果。

> [!TIP]
> 第一次运行项目可能会遇到找不到模块的错误，一般可以通过执行
> ```bash
> export PYTHONPATH=$PWD
> ```
> 解决

## 框架基本介绍

下面简单介绍一下这个基本框架。

### 数据

所有数据以图片形式放在项目根目录的 `data` 目录下， `data` 目录下有两个子目录， `processed` 存放预处理后的数据（一般不用）， `raw` 存放原来的图片， `raw` 目录下有两个子目录 `content` 和 `style` 目录，分别存放内容图片和风格图片。下载仓库之后会发现其中已经有了两张图片，作为基本的训练数据，来自 [Pytorch教程](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) 。

### 代码

所有实验相关的代码存放在 `style_transfer` 目录下，并有 `__init__.py` 使其成为一个包。

#### 数据处理

目录下 `data.py` 负责数据处理相关工作，框架仅仅封装了基本的操作，用户需要自行组合。

#### 训练和评估

目录下 `train.py` 和 `evaluate.py` 分别负责训练和评估模型，训练部分的代码基本完成，但有些许需要改进，评估部分尚未完成。训练过程需要配置较多的参数，为了实验的方便，框架采用读取配置文件的方式，相关的配置文件应该存放在 `experiments` 目录下，如 `config_gatys.json` ，使用是需要指定命令行参数 `config` 为对应的配置文件路径，你可以在配置文件中指定训练的参数、内容图像和风格图像的路径和图像保存路径，基本的样例参考[config_gatys.json](./experiments/config_gatys.json)。

#### 支持函数

目录下 `utils` 子目录存放一些常用的支持函数，作为子模块。其中 `func_utils.py` 实现了一些常用的函数工具， `json_loader.py` 封装了 `Json` 文件加载的功能， `metrices.py` 用于封装一些度量效果的函数，但尚未完成， `visualization.py` 封装了一些关于可视化的函数。

#### 模型定义

模型定义为一个子模块，在 `models` 目录下， `feature_extractor.py` 定义了特征提取器，包含一个抽象类 `FeatureExtractor` ，你可以继承这个类实现自己的特征提取器，作为示例，框架实现了一个基于 `CNN` 的特征提取器，主要用预训练的 `VGG-19` 完成。 `neural_style_transfer.py` 封装了神经风格迁移的抽象类 `NeuralStyleTransferModel` ，用户需要重载 `forward` 方法，作为示例，我们在 `gatys.py` 中实现了 `GatysStyleTransferModel` 类，算法主要参考了[这篇论文](https://dl.acm.org/doi/10.1145/3123266.3123425)。为了灵活的添加新的功能，我们采用装饰器模式，在 `neural_style_transfer_decorator.py` 中定义了神经风格迁移的装饰器，用户可以通过重载 `forward` 方法添加功能。作为示例，我们在 `lapstyle.py` 中实现了 `LapStyleTransferModel` 类，算法主要参考了[这篇论文](https://ieeexplore.ieee.org/document/7780634)。

### 测试

我们在 `tests` 目录下定义了测试模块，所有单元测试文件应该放在这个目录下，目前实现了 `test_data.py` ， `test_json.py` 和 `test_visualization.py` ，你可以根据需要自行定义其他测试。测试采用 [pytest](https://docs.pytest.org/en/stable/index.html) 框架，可以通过运行

```bash
pytest
```

运行所有的测试。

> [!IMPORTANT]
> 所有提交都需要确保通过已有的测试，新增的功能最好也要进行自测，并增加必要的测试文件。

### 其他事项

#### 代码风格

框架采用 [Google 风格](https://google.github.io/styleguide/pyguide.html) ，使用 `black` 进行格式化，使用 `pylint` 进行风格检查，这两个工具应该已经在[环境搭建](#环境搭建)的时候安装了。使用方法为

```bash
black file.py
pylint file.py
```

> [!IMPORTANT]
> 所有提交的文件都必须进行格式化并通过 pylint 的检查，大部分的检查不通过都是代码风格或安全性的问题，应该修改以通过检查，对于确信代码没问题或暂无办法解决，认为不会有太大影响的，可以局部禁用 pylint 检查，但应该通过注释说明禁用理由。

#### 自动化脚本

为了使用的方便，项目提供了 `Makefile` ，定义了一些目标，可以自动执行一些命令，以下是基本介绍：

1. test 目标：运行测试
2. lint 目标：使用 pylint 对代码进行静态检查和分析
3. format 目标：使用 black 对代码进行格式化
4. clean 目标：清理缓存文件
5. gatys 目标：读入 experiments/config_gatys.json 配置文件进行训练
6. lapstyle 目标：读入 experiments/config_lapstyle.json 配置文件进行训练

可以通过

```bash
make <target_name>
```

使用。注意你需要安装有 [GNU Make](https://www.gnu.org/software/make/) ，并有一些目标可能需要在类 Unix 系统才能正常构建。

> [!TIP]
> 这一部分不是必须的，仅仅是为了开发的方便。

## 开发

### 完善或修补框架

框架存在一些问题，已发现的已经标注出 `TODO` 或者 `FIXME` ，但还有其他未发现的问题，开发问题中遇到框架问题可以自由的修补或增加新的模块，但注意进行测试并通过已有的测试。

### 基于框架开发

框架提供了较好的拓展，可以通过重载 `FeatureExtractor` 类实现自己的特征提取器，重载 `NeuralStyleTransferModel` 和 `NeuralStyleTransferDecorator` 等实现自己的风格迁移模型，也可以自由的开发自己的模型等。可以对训练进行提高，比如使用学习率调度，使用更好的优化器等，但建议封装新的模块，以免训练代码过于冗长。

### 其他问题

#### 在自己的分支开发

建议将项目下载到本地后创建新的分支，以自己的用户名命名，推送到远程仓库，在自己的分支进行开发，待自己开发的部分基本稳定后发起 Pull Request 合并到主分支。

我开启了 Pull Request 的要求，但这是私有仓库，所以没有强制性；开发者都是仓库所有者，具有直接修改主分支的权限，但这样可能导致一些代码冲突，最好不要这样做。

在提交或推送之前，应该注意拉取最新的主分支，减少代码冲突。

> [!IMPORTANT]
> 不要强制推送、合并，涉及冲突的代码应该与对应的开发者讨论合并方式。

#### 注意同步环境

最好创建一个新的 Anaconda 环境，并仅安装[环境搭建](#环境搭建)中提到的必要的包，如果需要新的包，安装后应该向其他同学说明，或在 Pull Request 中说明，以便大家更新环境。

## 基于前馈网络的代码框架

基于前馈网络的代码实现主要依照[论文](https://arxiv.org/abs/1603.08155)，参考了 `Pytorch` [样例仓库](https://github.com/pytorch/examples/tree/main) 的[代码](https://github.com/pytorch/examples/tree/main/fast_neural_style)。

### 数据

训练数据暂定使用 [COCO](https://cocodataset.org/) 数据集，使用2017年的数据，将下载到 [data/coco](data/coco/) 目录下，为了使用的方便，框架提供了下载脚本，Linux平台可以使用[get_coco.sh](./get_coco.sh)脚本，其他平台可以使用[get_coco.py](./get_coco.py)脚本。

> [!NOTE]
> 暂时用的是 [COCO](https://cocodataset.org/) 数据集，这个数据集很大，可能需要下载很久，大家也可以找一下或许有更好的数据集

### 基本组成

框架在[transfer_net.py](./style_transfer/models/transfer_net.py)定义了生成网络，在[loss_net.py](./style_transfer/models/loss_net.py)定义了损失网络，在[dataset.py](./style_transfer/dataset.py)定义了 `COCO` 数据集，定义了训练模块[fast_train.py](./style_transfer/train.py)，评估模块[fast_evaluate](./style_transfer/fast_evaluate.py)和应用模块[fast_stylize.py](./style_transfer/fast_stylize.py)。除了损失网络、训练、数据集基本完成外，其他模块均未完成。

配置文件为[config_fast.json](./experiments/config_fast.json)，但尚不完善。

### 需要完成的部分

#### 完善框架

当前的框架并不完整，许多内容尚未完成，大家需要找到标记了 `TODO` 的地方进行补充。许多逻辑上不完善，可能在完成一个部分的时候需要改动其他部分、增加一些内容。

实现的时候可以参考Pytorch的[实现]((https://github.com/pytorch/examples/tree/main/fast_neural_style))，注意稍微修改一下。

比较大的几个需要完成的模块是

- [ ] 转换网络
- [ ] 训练模块的预处理
- [ ] 评估模块，主要就是加载模型，在验证集或测试集上计算损失，或者你也可以开发更好的评估方式
- [ ] 应用模块，主要为提供接口给用户，输入一张图片，输出一张风格迁移后的图片
- [ ] 调试，这个框架没有经过实验，即使完成了所有模块也可能不能正常运行，需要调试

#### 适配训练

训练需要很多计算，笔记本不一定能完成，可能需要使用云服务，所以可能需要提供 `ipynb` 文件，这部分工作可以待框架完成后再行整合。
