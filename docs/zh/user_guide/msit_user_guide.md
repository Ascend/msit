# MindStudio Inference Tools 使用指南

<br>

MindStudio Inference Tools（msIT）面向昇腾 AI 模型推理开发中的关键挑战，包含模型压缩、调试与调优等工具。

## 工具场景化推荐

本节以“我要做什么”为导向，旨在解决“当前问题应选用哪个工具”的困惑，结合典型场景为您推荐合适的工具，请点击对应工具链接跳转至其代码仓库，深入了解使用方法：

| 我要做什么 | 推荐工具 | 为什么推荐 |
|-----------|----------|------------|
| 我有一个开源模型，想生成量化权重 | [msModelSlim](https://gitcode.com/Ascend/msmodelslim) | 支持一键量化功能，提供量化模型的最佳实践库，快速生成量化权重 |
| 我量化后精度掉太多，想"自动找一套能达标的方案" | [msModelSlim](https://gitcode.com/Ascend/msmodelslim) | 给出精度目标即可，工具自动迭代：生成 yaml → 量化 → vLLM-Ascend 拉起服务 → AISBench 评估 → 不达标再换一套，直到命中目标。 |
| 我在量化精度调优过程中，不知道该回退哪些层 | [msModelSlim](https://gitcode.com/Ascend/msmodelslim) | 提供多种粒度，给出对量化敏感层的建议 |
| 我只想做权重格式/精度转换 | [msModelSlim](https://gitcode.com/Ascend/msmodelslim) | 不加载模型代码、不需要校准集、CPU可跑的快速完成权重格式转换（如 BF16 转 MXFP8） |
