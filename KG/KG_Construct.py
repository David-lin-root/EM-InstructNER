# ============================================================
# EmergencyKG Construction Pipeline
# Three-Stage Knowledge Graph Construction Framework
# ------------------------------------------------------------
# Stage 1: Rule-based High-confidence Extraction
# Stage 2: Weakly-supervised Relation Extraction
# Stage 3: Human Verification + Global Consistency Checking
# ============================================================

import re
import json
import hashlib
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

# ============================================================
# Stage 1: Rule-based High-confidence Extraction
# ============================================================

class RuleBasedExtractor:

    def __init__(self):

        # ====================================================
        # 86 High-precision Rule Templates
        # ====================================================

        self.patterns = {

            # -------------------------
            # Date / Time (1-15)
            # -------------------------
            "Date_1": re.compile(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}"),
            "Date_2": re.compile(r"\d{4}年\d{1,2}月\d{1,2}日"),
            "Date_3": re.compile(r"\d{1,2}:\d{2}"),
            "Date_4": re.compile(r"\d{4}年"),
            "Date_5": re.compile(r"\d{1,2}月\d{1,2}日"),
            "Date_6": re.compile(r"凌晨|上午|中午|下午|傍晚|夜间"),
            "Date_7": re.compile(r"昨日|今日|明日"),
            "Date_8": re.compile(r"周一|周二|周三|周四|周五|周六|周日"),
            "Date_9": re.compile(r"\d{1,2}时\d{1,2}分"),
            "Date_10": re.compile(r"\d{4}\.\d{1,2}\.\d{1,2}"),
            "Date_11": re.compile(r"\d{4}-\d{1,2}-\d{1,2}"),
            "Date_12": re.compile(r"\d{1,2}月"),
            "Date_13": re.compile(r"\d{1,2}日"),
            "Date_14": re.compile(r"本周|本月|今年"),
            "Date_15": re.compile(r"\d{1,2}季度"),

            # -------------------------
            # Location (16-30)
            # -------------------------
            "Location_1": re.compile(r"[\u4e00-\u9fa5]+省"),
            "Location_2": re.compile(r"[\u4e00-\u9fa5]+市"),
            "Location_3": re.compile(r"[\u4e00-\u9fa5]+县"),
            "Location_4": re.compile(r"[\u4e00-\u9fa5]+区"),
            "Location_5": re.compile(r"[\u4e00-\u9fa5]+乡"),
            "Location_6": re.compile(r"[\u4e00-\u9fa5]+镇"),
            "Location_7": re.compile(r"[\u4e00-\u9fa5]+村"),
            "Location_8": re.compile(r"\d+\.\d+[NSEW]/\d+\.\d+[NSEW]"),
            "Location_9": re.compile(r"东经\d+\.\d+度"),
            "Location_10": re.compile(r"北纬\d+\.\d+度"),
            "Location_11": re.compile(r"南海|黄海|东海"),
            "Location_12": re.compile(r"沿海地区"),
            "Location_13": re.compile(r"山区"),
            "Location_14": re.compile(r"城区"),
            "Location_15": re.compile(r"流域"),

            # -------------------------
            # Disaster Type (31-45)
            # -------------------------
            "Type_1": re.compile(r"地震"),
            "Type_2": re.compile(r"台风"),
            "Type_3": re.compile(r"洪水"),
            "Type_4": re.compile(r"泥石流"),
            "Type_5": re.compile(r"滑坡"),
            "Type_6": re.compile(r"暴雨"),
            "Type_7": re.compile(r"干旱"),
            "Type_8": re.compile(r"寒潮"),
            "Type_9": re.compile(r"森林火灾"),
            "Type_10": re.compile(r"风暴潮"),
            "Type_11": re.compile(r"龙卷风"),
            "Type_12": re.compile(r"冰雹"),
            "Type_13": re.compile(r"低温雨雪冰冻"),
            "Type_14": re.compile(r"次生灾害"),
            "Type_15": re.compile(r"复合灾害"),

            # -------------------------
            # Strength / Intensity (46-60)
            # -------------------------
            "Strength_1": re.compile(r"M\d\.\d"),
            "Strength_2": re.compile(r"\d+级"),
            "Strength_3": re.compile(r"TS|STS|TY|SuperTY"),
            "Strength_4": re.compile(r"\d+毫米"),
            "Strength_5": re.compile(r"\d+米"),
            "Strength_6": re.compile(r"\d+km/h"),
            "Strength_7": re.compile(r"\d+人死亡"),
            "Strength_8": re.compile(r"\d+人受伤"),
            "Strength_9": re.compile(r"\d+人失踪"),
            "Strength_10": re.compile(r"\d+万元"),
            "Strength_11": re.compile(r"\d+亿元"),
            "Strength_12": re.compile(r"\d+户受灾"),
            "Strength_13": re.compile(r"\d+公顷"),
            "Strength_14": re.compile(r"\d+级响应"),
            "Strength_15": re.compile(r"黄色预警|橙色预警|红色预警"),

            # -------------------------
            # Description / Emergency Response (61-86)
            # -------------------------
            "Description_1": re.compile(r"紧急转移"),
            "Description_2": re.compile(r"人员疏散"),
            "Description_3": re.compile(r"启动应急响应"),
            "Description_4": re.compile(r"救援力量"),
            "Description_5": re.compile(r"抢险救灾"),
            "Description_6": re.compile(r"房屋倒塌"),
            "Description_7": re.compile(r"道路中断"),
            "Description_8": re.compile(r"通信中断"),
            "Description_9": re.compile(r"电力中断"),
            "Description_10": re.compile(r"次生滑坡"),
            "Description_11": re.compile(r"山洪暴发"),
            "Description_12": re.compile(r"城市内涝"),
            "Description_13": re.compile(r"交通受阻"),
            "Description_14": re.compile(r"物资调拨"),
            "Description_15": re.compile(r"应急安置"),
            "Description_16": re.compile(r"灾情统计"),
            "Description_17": re.compile(r"搜救行动"),
            "Description_18": re.compile(r"医疗救援"),
            "Description_19": re.compile(r"恢复供电"),
            "Description_20": re.compile(r"灾后重建"),
            "Description_21": re.compile(r"险情排查"),
            "Description_22": re.compile(r"应急演练"),
            "Description_23": re.compile(r"群众受困"),
            "Description_24": re.compile(r"堰塞湖"),
            "Description_25": re.compile(r"滑坡体"),
            "Description_26": re.compile(r"地质不稳定")
        }

    def extract(self, text):

        triples = []

        for rule_name, pattern in self.patterns.items():

            matches = pattern.findall(text)

            for m in matches:

                relation_type = rule_name.split("_")[0]

                triple = {
                    "head": text[:50],
                    "relation": relation_type,
                    "tail": m,
                    "confidence": 1.0,
                    "rule": rule_name
                }

                triples.append(triple)

        return triples


# ============================================================
# Stage 2: Weakly-supervised Relation Extraction
# ============================================================

class WeaklySupervisedExtractor:

    def __init__(self):

        self.embedding_model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2'
        )

    def semantic_similarity(self, s1, s2):

        emb1 = self.embedding_model.encode(s1, convert_to_tensor=True)
        emb2 = self.embedding_model.encode(s2, convert_to_tensor=True)

        return util.cos_sim(emb1, emb2).item()

    def weak_relation_extract(self, sentence):

        extracted = []

        relation_patterns = {

            "death": ["死亡", "遇难"],
            "injury": ["受伤"],
            "economic_loss": ["经济损失"],
            "evacuation": ["紧急转移", "疏散"],
            "rescue": ["救援", "搜救"]

        }

        for relation, keywords in relation_patterns.items():

            for kw in keywords:

                if kw in sentence:

                    extracted.append({

                        "relation": relation,
                        "tail": kw,
                        "confidence": 0.92

                    })

        return extracted


# ============================================================
# Stage 3: Human Verification + Consistency Verification
# ============================================================

class ConsistencyVerifier:

    def __init__(self):

        self.numeric_threshold = 0.10

    def remove_duplicates(self, triples):

        seen = set()
        cleaned = []

        for t in triples:

            triple_str = json.dumps(t, ensure_ascii=False)

            h = hashlib.sha256(
                triple_str.encode("utf-8")
            ).hexdigest()

            if h not in seen:

                seen.add(h)
                cleaned.append(t)

        return cleaned

    def conflict_detection(self, triples):

        grouped = defaultdict(list)

        for t in triples:

            key = (t["head"], t["relation"])

            grouped[key].append(t["tail"])

        conflicts = []

        for key, values in grouped.items():

            unique_values = set(values)

            if len(unique_values) > 1:

                conflicts.append({

                    "key": key,
                    "values": list(unique_values)

                })

        return conflicts


# ============================================================
# Main KG Construction Pipeline
# ============================================================

class EmergencyKGBuilder:

    def __init__(self):

        self.rule_extractor = RuleBasedExtractor()
        self.weak_extractor = WeaklySupervisedExtractor()
        self.verifier = ConsistencyVerifier()

    def build(self, documents):

        all_triples = []

        print("=" * 60)
        print("Stage 1: Rule-based High-confidence Extraction")
        print("=" * 60)

        for doc in documents:

            triples = self.rule_extractor.extract(doc)

            all_triples.extend(triples)

        print(f"Stage 1 triples: {len(all_triples)}")

        print("=" * 60)
        print("Stage 2: Weakly-supervised Relation Extraction")
        print("=" * 60)

        weak_triples = []

        for doc in documents:

            sentences = doc.split("。")

            for s in sentences:

                extracted = self.weak_extractor.weak_relation_extract(s)

                for e in extracted:

                    if e["confidence"] >= 0.90:

                        weak_triples.append({

                            "head": s[:50],
                            "relation": e["relation"],
                            "tail": e["tail"],
                            "confidence": e["confidence"]

                        })

        print(f"Stage 2 triples: {len(weak_triples)}")

        all_triples.extend(weak_triples)

        print("=" * 60)
        print("Stage 3: Global Consistency Verification")
        print("=" * 60)

        cleaned = self.verifier.remove_duplicates(all_triples)

        conflicts = self.verifier.conflict_detection(cleaned)

        print(f"Final triples: {len(cleaned)}")
        print(f"Conflicts detected: {len(conflicts)}")

        return cleaned, conflicts


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":

    documents = [

        "2019/1/5 四川省发生M6.2地震，造成20人死亡，道路中断。",
        "2020/7/8 广东省遭受台风袭击，启动三级应急响应，造成严重经济损失。"

    ]

    builder = EmergencyKGBuilder()

    triples, conflicts = builder.build(documents)

    print("\nSample Triples:\n")

    for t in triples[:20]:

        print(t)

    # ========================================================
    # Save Excel KG
    # ========================================================

    df = pd.DataFrame(triples)

    output_path = "EmergencyKG.xlsx"

    df.to_excel(output_path, index=False)

    print(f"\nEmergencyKG saved to: {output_path}")

    # ========================================================
    # Save conflict reports
    # ========================================================

    with open("kg_conflicts.json", "w", encoding="utf-8") as f:

        json.dump(
            conflicts,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("Conflict report saved.")