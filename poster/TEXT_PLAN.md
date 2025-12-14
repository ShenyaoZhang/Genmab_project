# Poster文字部分规划

## 总体要求
- 总字数：**800字以内**
- 风格：简洁、专业、突出方法和结果
- 重点：用图和表展示，文字辅助说明

---

## 字数分配方案

### 1. **Title** (标题)
- **建议标题**：`AI-Powered Pharmacovigilance: Detecting Rare Adverse Events in Oncology Drugs`
- 或：`Machine Learning Pipeline for Rare & Unexpected Adverse Event Detection`

### 2. **Introduction** (~80-100 words)
**内容要点**：
- 背景：药物安全监测的重要性，特别是肿瘤药物
- 问题：如何从大量FAERS数据中发现罕见和意外的药物-不良事件关系
- 目标：开发自动化检测系统，识别需要关注的信号
- 方法概述：结合NLP、异常检测、生存分析和严重程度预测

**示例文字**：
```
Pharmacovigilance plays a critical role in drug safety monitoring, especially for oncology therapeutics where rare adverse events (AEs) may emerge post-market. We developed an integrated machine learning pipeline to automatically detect rare and unexpected drug-AE relationships from FDA Adverse Event Reporting System (FAERS) data. Our approach combines natural language processing, anomaly detection, survival analysis, and severity prediction to identify safety signals that warrant clinical attention.
```

### 3. **Task 1: NLP-Based AE Extraction** (~80-100 words)
**标题**：`NLP-Based Adverse Event Extraction`

**内容要点**：
- 方法：使用BERT/ClinicalBERT从非结构化临床文本中提取AE
- 技术：命名实体识别(NER) + MedDRA术语映射
- 输出：结构化AE数据
- 应用：为后续分析提供高质量特征

**示例文字**：
```
We employ BERT-based natural language processing to extract adverse events from unstructured clinical narratives in FAERS reports. Our pipeline uses ClinicalBERT for medical entity recognition and maps extracted terms to MedDRA standardized terminology. This enables automated extraction of AEs from free-text fields, enriching structured data with information that would otherwise require manual review.
```

### 4. **Task 2: Survival Analysis & Risk Factors** (~80-100 words)
**标题**：`Time-to-Event Analysis & Risk Stratification`

**内容要点**：
- 方法：Cox比例风险模型进行生存分析
- 目标：识别影响AE发生时间的风险因素
- 输出：风险分层模型和特征重要性
- 验证：结果与临床试验一致

**示例文字**：
```
Survival analysis using Cox proportional hazards models identifies risk factors that influence time-to-adverse-event occurrence. We analyze patient demographics, medical history, and concomitant medications to stratify patients by risk level. Feature importance analysis provides interpretable insights, with results validated against clinical trial data (EPCORE NHL-1).
```

### 5. **Task 3: Rare & Unexpected AE Detection** (~100-120 words)
**标题**：`Rare & Unexpected Signal Detection`

**内容要点**：
- 方法：Isolation Forest异常检测 + 4步过滤流程
- 流程：Isolation Forest → 移除已知AE → 移除适应症术语 → 移除高频AE
- 统计验证：PRR, IC025, Chi-square
- 结果：识别1386个罕见意外信号
- 临床特征分析：BERT分析风险因子

**示例文字**：
```
We detect rare and unexpected drug-AE relationships using Isolation Forest anomaly detection combined with a 4-step filtering pipeline: (1) statistical anomaly identification, (2) FDA label filtering, (3) indication term removal, and (4) frequency-based filtering. Results are validated using disproportionality metrics (PRR>2, IC025>0, Chi-square>4). Our pipeline identified 1,386 rare signals across 37 oncology drugs. For each signal, BERT-based clinical feature analysis identifies demographic and medical history risk factors.
```

### 6. **Task 5: Severity Prediction** (~80-100 words)
**标题**：`Adverse Event Severity Prediction`

**内容要点**：
- 方法：随机森林分类器
- 特征：患者特征、药物信息、AE类型
- 输出：预测AE严重程度（死亡、住院、危及生命等）
- 应用：辅助临床决策

**示例文字**：
```
A Random Forest classifier predicts adverse event severity (death, hospitalization, life-threatening) based on patient demographics, drug characteristics, and event type. The model enables early identification of high-risk cases, supporting clinical decision-making and resource allocation for patient monitoring.
```

### 7. **Conclusion & Future Work** (~100 words)
**内容要点**：
- 总结：集成系统实现了端到端的药物安全监测
- 贡献：自动化检测罕见信号，提供可解释的风险分析
- 未来方向：实时监测、多数据源整合、因果推断增强

**示例文字**：
```
Our integrated pipeline enables end-to-end pharmacovigilance analysis, from unstructured text extraction to rare signal detection and risk stratification. The system automatically identifies 1,386 rare signals and provides interpretable risk factor analysis for clinical review. Future work includes real-time monitoring capabilities, integration of multiple data sources (EudraVigilance, JADER), and enhanced causal inference methods to distinguish causal risk factors from correlations.
```

### 8. **Key Results** (~50-80 words)
**要点列表**：
- 检测到1,386个罕见意外信号
- 覆盖37种肿瘤药物
- 统计验证通过率89.1%
- 临床特征分析识别关键风险因子
- 结果与临床试验数据一致

**示例文字**：
```
• Detected 1,386 rare & unexpected drug-AE signals across 37 oncology drugs
• Statistical validation: 89.1% pass PRR, IC025, and Chi-square thresholds
• BERT clinical analysis identifies demographic and medical history risk factors
• Survival analysis results validated against clinical trial data
• Automated pipeline enables scalable pharmacovigilance monitoring
```

---

## 总字数统计
- Introduction: ~90 words
- Task 1: ~90 words
- Task 2: ~90 words
- Task 3: ~110 words
- Task 5: ~90 words
- Conclusion: ~100 words
- Key Results: ~60 words
- **总计：~630 words** (在800字限制内)

---

## 实施建议

1. **保持简洁**：每个部分聚焦核心方法和结果
2. **突出数字**：强调具体成果（1386个信号、37种药物等）
3. **视觉优先**：文字配合图表，不重复图表信息
4. **专业术语**：使用标准医学术语和统计学术语
5. **可读性**：使用短句和清晰的段落结构


