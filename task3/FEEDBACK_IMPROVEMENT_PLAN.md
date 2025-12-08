# Task 3 Feedback Improvement Plan

## 概述
根据Genmab团队的feedback，分析哪些改进点属于Task3的范围，哪些不属于。

---

## ✅ 属于Task3范围的改进点

### 1. **Clarify rare/unexpected AE detection steps** ⭐⭐⭐ (最高优先级)

**当前状态：**
- 代码中有完整的过滤逻辑（`task3_drug_label_filter.py`）
- 但缺少清晰的文档说明和流程图

**需要改进：**
- ✅ 添加流程图：
  ```
  All AE pairs 
    → Remove known label AEs (FDA drug labels)
    → Remove indication-related terms
    → Remove high-frequency AEs (count >= mean)
    → Flag remaining rare unexpected AEs
  ```
- ✅ 添加具体例子：
  - "epcoritamab + renal impairment appeared only twice, is not on the drug label, and is below frequency thresholds (mean=3.24), so it is flagged as unexpected."
- ✅ 在README中清晰说明每一步的过滤逻辑

**实施方式：**
- 更新 `README.md`，添加详细的流程图（文本或ASCII art）
- 添加 `DETECTION_STEPS.md` 文档，包含具体例子
- 在代码注释中补充说明

---

### 2. **Show how an end-user interacts with the tool** ⭐⭐ (高优先级)

**当前状态：**
- 有 `task3_interactive_query.py` 交互查询系统
- 但缺少清晰的使用示例和UI mock

**需要改进：**
- ✅ 提供更清晰的使用示例：
  ```python
  from task3_interactive_query import InteractiveAnomalyQuery
  query = InteractiveAnomalyQuery()
  result = query.check_any_combo("epcoritamab", "neutropenia")
  # Output: "RARE & UNEXPECTED" or "NOT RARE/UNEXPECTED"
  ```
- ✅ 添加简单的UI mock（PPT中的框图和流程图）：
  ```
  [Dropdown: Drug] → [Dropdown: Adverse Event] → [Run Button]
                                                      ↓
                                              [Output Panel]
                                              - Risk Score
                                              - Top Features
                                              - Statistical Metrics
  ```
- ✅ 展示输出格式示例：
  ```
  Drug: Epcoritamab
  Adverse Event: Neutropenia
  Status: RARE & UNEXPECTED
  Observed in: FAERS
  Count: 2
  PRR: 15.3
  IC025: 2.1
  Chi-square: 8.5
  ```

**实施方式：**
- 更新 `README.md`，添加详细的使用示例
- 创建 `USAGE_EXAMPLES.md`，包含多个实际例子
- 在PPT中添加UI mock图

---

### 3. **Improve interpretability of the models** ⭐ (中优先级，部分相关)

**当前状态：**
- BERT临床特征分析（`task3_bert_clinical_features.py`）有SHAP输出
- 但缺少对SHAP/feature importance的简单解释

**需要改进：**
- ✅ 添加SHAP/feature importance的简单解释：
  - "A positive SHAP value means the feature pushes risk upward. For example, age > 65 increased the predicted Neutropenia risk for Patient A."
  - "A negative SHAP value means the feature reduces risk. For example, female sex reduced the predicted risk."
- ✅ 添加模型用途表格：
  | Model | Purpose |
  |-------|---------|
  | Isolation Forest (Rare AE Model) | Detects unexpected AE patterns that are statistically rare |
  | BERT Clinical Features | Identifies clinical risk factors (age, sex, medical history) that influence specific AEs |

**实施方式：**
- 在 `task3_bert_clinical_features.py` 的输出中添加解释性文本
- 创建 `MODEL_INTERPRETATION.md` 文档

---

### 4. **Add dataset and model summaries** ⭐ (中优先级，部分相关)

**当前状态：**
- 缺少数据摘要和missingness summary

**需要改进：**
- ✅ 添加FAERS数据摘要：
  - Total drug-event pairs: X
  - Number of rare unexpected AE cases: Y
  - Percentage of complete cases: Z%
- ✅ 添加missingness summary table：
  | Variable | Missing % | Notes |
  |----------|-----------|-------|
  | PRR | 0% | Calculated from contingency table |
  | IC025 | 0% | Calculated from contingency table |
  | Count | 0% | Direct count from FAERS |
  | Drug Name | 0% | From FAERS reports |
  | Event Name | 0% | From FAERS reports |
- ⚠️ **注意**：Task3只用FAERS，不涉及JADER和EV

**实施方式：**
- 在 `task3_improved_pipeline.py` 中添加数据摘要输出
- 创建 `DATA_SUMMARY.md` 文档

---

### 5. **List all variables used and their data sources** ⭐ (低优先级)

**当前状态：**
- 缺少变量清单

**需要改进：**
- ✅ 列出所有使用的特征变量：
  | Variable | Type | Data Source | Description |
  |----------|------|-------------|--------------|
  | PRR | Continuous | Calculated | Proportional Reporting Ratio |
  | IC025 | Continuous | Calculated | Information Component lower bound |
  | Chi-square | Continuous | Calculated | Chi-square statistic |
  | Count | Integer | FAERS | Number of reports for drug-event pair |
  | Drug Name | Categorical | FAERS | Drug name from reports |
  | Event Name | Categorical | FAERS | Adverse event name from reports |
- ✅ 说明数据来源：FAERS (FDA Adverse Event Reporting System)
- ⚠️ **注意**：Task3主要用统计特征，不是临床变量（临床变量在BERT分析中）

**实施方式：**
- 创建 `VARIABLES_INVENTORY.md` 文档
- 在 `README.md` 中添加变量说明

---

## ❌ 不属于Task3范围的改进点

### 1. **Make the full pipeline scalable**
- **原因**：用户明确说不管了，因为不连所有任务，各自弄各自的指令
- **Task3现状**：已经有独立的pipeline，可以单独运行

### 2. **Strengthen the polypharmacy analysis**
- **原因**：Task3主要检测rare和unexpected AE，不专门分析polypharmacy
- **Task3现状**：BERT临床特征分析中有concomitant drugs，但不是核心功能

### 3. **Demonstrate handling of continuous and ordinal variables**
- **原因**：Task3主要用统计特征（PRR, IC025等），不是临床连续变量
- **Task3现状**：统计特征都是计算得出的，不需要特殊处理

### 4. **Address missing clinical variables (e.g., disease stage)**
- **原因**：Task3不做临床变量分析，主要关注统计异常检测
- **Task3现状**：BERT分析中有一些临床特征，但disease stage不在FAERS中

### 5. **Provide database-specific analyses**
- **原因**：Task3只用FAERS，不涉及JADER和EV
- **Task3现状**：数据来源单一，无法做跨数据库比较

### 6. **Improve causal inference and time-to-event explanations**
- **原因**：Task3不做causal inference，主要做异常检测
- **Task3现状**：Isolation Forest是unsupervised anomaly detection，不是causal model

### 7. **Biomarker integration**
- **原因**：Task3不涉及biomarker分析
- **Task3现状**：主要关注drug-event关系的统计异常，不涉及biomarker数据

---

## 📋 实施优先级和时间安排

### Phase 1: 核心功能文档化（必须完成）
1. ✅ Clarify rare/unexpected AE detection steps
   - 时间：2-3小时
   - 产出：更新README + 创建DETECTION_STEPS.md

2. ✅ Show how an end-user interacts with the tool
   - 时间：1-2小时
   - 产出：更新README + 创建USAGE_EXAMPLES.md + PPT UI mock

### Phase 2: 可选改进（有时间就做）
3. ⚠️ Improve interpretability of the models
   - 时间：1-2小时
   - 产出：更新BERT输出 + 创建MODEL_INTERPRETATION.md

4. ⚠️ Add dataset and model summaries
   - 时间：1-2小时
   - 产出：添加数据摘要输出 + 创建DATA_SUMMARY.md

5. ⚠️ List all variables used and their data sources
   - 时间：1小时
   - 产出：创建VARIABLES_INVENTORY.md

---

## 🎯 总结

**必须完成的改进（2项）：**
1. Clarify rare/unexpected AE detection steps
2. Show how an end-user interacts with the tool

**可选完成的改进（3项）：**
3. Improve interpretability of the models
4. Add dataset and model summaries
5. List all variables used and their data sources

**不需要做的改进（7项）：**
- 都与Task3的核心功能无关，属于其他task的范围

---

## 📝 下一步行动

1. 先完成Phase 1的两个必须改进项
2. 根据时间情况，选择性完成Phase 2的改进项
3. 所有改进都要更新到GitHub仓库的README和文档中

