# 基于MARL的仓储物流路径规划与任务分配系统

## 描述
设计一个多智能体强化学习（MARL）系统，实现仓储物流中多台自动导引车的协同路径规划与任务分配，以最小化货物运输时间、避免碰撞并优化资源利用率。

## 任务
1. 环境建模与构建
2. MARL算法实现
3. 策略可视化分析

## 训练优化指标
最小化货物平均运输时间和碰撞次数

## 环境配置
```bash
conda create -n warehouse python=3.10
conda activate warehouse
conda install -c conda-forge libstdcxx-ng
pip install -r requirements.txt
```