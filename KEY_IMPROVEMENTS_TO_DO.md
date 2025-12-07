# 关键改进清单（对照 Sky 的要求）

## 已满足的点 ✅

1. ✅ CRS 专项死亡模型 + granular 分析
2. ✅ 模型可解释性（SHAP）
3. ✅ Polypharmacy 和合并症分析
4. ✅ 连续变量的处理和可视化

## 必须补的关键点

### 1. Cancer Stage 真正用起来（Nicole 显式点名）⭐ 最高优先级

需要在以下文件中补充：

- **11_granular_crs_analysis.py**
  - 在 `stratified_analysis()` 中加入 stage 分层的死亡率统计
  - 在 `granular_feature_engineering()` 中确保从 `drug_indication` 提取 stage 信息
  - 生成 stage 分层的可视化图表

- **granular_crs_report.md**
  - 加入 "Cancer Stage and CRS-related Death" 章节
  - 包含 Stage I-II vs Stage III-IV 的死亡率对比

- **13_crs_shap_analysis.py**
  - 在 plain language summary 中加入 stage 的解释（已有 mapping，需要确保在 summary 中出现）

### 2. Pipeline wrapper 函数

- 创建 `run_crs_mortality_pipeline(drug, adverse_event)` wrapper
- 可以放在 `12_crs_model_training.py` 中（已有基础，需要完善）或新建 util 文件

### 3. 数据和缺失值说明

- 检查 `11_granular_crs_analysis.py` 中的 `generate_missingness_summary()` 是否完整
- 在报告中加入缺失值说明

---

## 实施计划

按照优先级顺序：

1. **立即完成**：在 granular 分析中加入 stage 分层统计
2. **尽快完成**：Pipeline wrapper 函数
3. **检查完善**：缺失值说明和报告更新

