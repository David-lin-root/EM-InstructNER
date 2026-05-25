# EM-InstructNER: 自然灾害应急事件命名实体识别指令微调数据集与框架

本仓库为 **EM-InstructNER** 项目部分开源内容，致力于解决自然灾害应急领域命名实体识别（NER）中的领域知识不足、新兴实体识别困难以及标注数据稀缺等问题。

---

## 项目亮点

- **EM-InstructNER**：面向自然灾害应急场景构建的高质量指令微调数据集
- **EmergencyKG**：领域专用灾害知识图谱（支持动态更新）
- **KG-RAG**：知识图谱增强检索机制，提升长尾与新兴实体识别能力
- **LoRA 高效微调**：基于 Qwen2-7B-Instruct 的参数高效微调方案
- 显著提升了灾害应急文本中 Date、Location、Type、Description、Strength 五类实体的识别性能

---

## 仓库结构

```bash
EM-InstructNER/
├── dataset/                  # 公开数据集（部分）
│   ├── test.json             # 1537条高质量指令数据
│   └── splits.json      
├── KG/                       # 知识图谱构建相关
│   ├── KG_Construct.py       # 三阶段KG构建脚本
│   └── KG-metadata.xlsx
├── prompt/                   # Prompt模板
│   ├── KG-RAG prompt.txt
│   └── ner_prompt.txt
├── scripts/                  # 训练与部署脚本
│   ├── lora_sft.sh           # LoRA微调主脚本
│   └── Merge_ck.sh           # 模型合并脚本
├── code/                     # 核心训练代码
│   └── run.py
├── requirements.txt
├── README.md
└── License.txt
---
🚀 快速开始
## 1. 环境安装
git clone https://github.com/David-lin-root/EM-InstructNER.git
cd EM-InstructNER
pip install -r requirements.txt
---
##2. LoRA 微调训练
cd scripts
bash lora_sft.sh
---
##3. 模型合并
bash Merge_ck.sh
---
##4. 推理测试
打开 code/test.ipynb 运行实体抽取测试。
---
📊 数据集说明
实体类型（5类）：
Date：日期和时间
Location：地理位置
Type：灾害类型
Description：灾害描述
Strength：灾害强度
---
公开规模：15,37 条高质量指令数据
完整规模：15,372 条
---
📈 核心方法
指令微调（Instruction Tuning）
知识图谱增强检索（KG-RAG）
LoRA 参数高效微调（训练参数仅为全参数的约1.2%）
三阶段知识图谱构建（规则 + 弱监督 + 人工校验）
---
##使用说明
Prompt 模板位于 prompt/ 目录
训练配置见 scripts/lora_sft.sh
知识图谱构建脚本：KG/KG_Construct.py
---
📄 引用
如果您使用了本项目，请引用我们的论文：
@article{zhang2026eminstructner,
  title={Named Entity Recognition Method for Natural Disaster Emergencies Based on Instruction Tuning and Graph Retrieval-Augmented Generation},
  author={Zhang Kehong and Lin Xinyu and Wang Min and others},
  journal={Big Data and Cognitive Computing},
  year={2026}
}
---
📜 License
本项目采用 MIT License 开源，详情见 License.txt。
主要用于学术研究，商业使用请联系作者。
##支持我们
如果本项目对您有帮助，欢迎 Star 本仓库！
如有问题或合作意向，欢迎提交 Issue。
