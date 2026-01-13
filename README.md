# AU3318 数字信号处理大作业 - 槽楔松紧度检测

本项目旨在利用数字信号处理技术，对发电机定子槽楔的敲击振动信号进行分析。通过对不同紧固压力下（400N - 4000N）采集的声学/振动信号进行处理，构建分类模型以识别槽楔的松紧状态，并探究松紧度与信号频率特征之间的定量关系。

## 项目结构

```
e:\sync\courses\AU3318\finalProj\
│
├── dsp_assignment_solution.py  # [核心代码] 完整的解决方案脚本（数据处理、特征提取、分类、绘图）
├── visualize.py               # 辅助脚本，用于简单的波形可视化和敲击次数估算
├── report/
│   └── report.tex             # 实验报告 LaTeX 源码
│
├── 槽楔模型测试数据/           # (需自行添加) 包含 acquisitionData-*.txt 的数据文件夹
│
├── .gitignore                 # Git 忽略配置
├── final_result_trend.png     # [结果] 松紧度与频率特征关系图
├── confusion_matrix.png       # [结果] 分类器混淆矩阵
├── spectrum_comparison.png    # [结果] 不同压力下的频谱对比图
├── waveform_with_peaks.png    # [过程] 时域波形与切分示意图
└── waveform_preview.png       # [过程] 原始数据波形预览
```

## 环境依赖

本项目使用 Python 3 开发，需要安装以下依赖库：

```bash
pip install numpy matplotlib scipy scikit-learn seaborn
```

## 运行方法

确保数据文件夹 `槽楔模型测试数据` 位于项目根目录下，然后运行核心脚本：

```bash
python dsp_assignment_solution.py
```

程序运行后将自动执行以下步骤：
1.  **数据读取与切分**：读取 TXT 数据，利用动态阈值法切分出单次敲击信号。
2.  **特征提取**：计算每个样本的频域特征（主频、质心、带宽等）和时域特征（峰度、RMS）。
    *   *注：为消除台架低频共振干扰，代码默认滤除了 <1500Hz 的信号。*
3.  **分类模型训练**：使用随机森林（Random Forest）进行分类，输出准确率和混淆矩阵。
4.  **结果可视化**：生成并保存所有分析图表。

## 关键结果

1.  **分类精度**：
    *   随机森林模型在测试集上的准确率达到 **86.96%**。
    *   对低压力（故障状态，如 400N）的识别准确率高达 **100%**。

2.  **物理规律**：
    *   **正相关性**：随着槽楔紧固压力（松紧度）的增加，信号的**频谱质心（Spectral Centroid）**呈现显著的线性上升趋势。
    *   这验证了振动理论公式 $f \propto \sqrt{k/m}$（接触刚度 $k$ 随压力增大而增大）。

## 实验报告

详细的实验原理、步骤分析及结论请参阅 `report/report.pdf`（需编译 `report.tex` 生成）。
