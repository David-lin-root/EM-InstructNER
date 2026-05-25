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

```text
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
└── License.txt
