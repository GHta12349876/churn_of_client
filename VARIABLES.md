# 变量说明文档

## 目录
- [常量配置](#常量配置)
- [数据预处理变量](#数据预处理变量)
- [模型特征变量](#模型特征变量)
- [模型训练与评估变量](#模型训练与评估变量)
- [阈值优化变量](#阈值优化变量)
- [风险分层变量](#风险分层变量)
- [服务类属性](#服务类属性)

---

## 常量配置

### 编码映射常量

| 变量名 | 类型 | 值 | 说明 |
|--------|------|-----|------|
| `INTERNET_MAP` | dict | `{'No':0, 'DSL':1, 'Fiber optic':2}` | 互联网服务类型的标签编码映射 |
| `CONTRACT_MAP` | dict | `{'Month-to-month':0, 'One year':1, 'Two year':2}` | 合同期限的标签编码映射 |

### 阈值配置常量

| 变量名 | 类型 | 值 | 说明 |
|--------|------|-----|------|
| `DEFAULT_THRESHOLD` | float | `0.35` | 默认分类阈值，用于判断客户是否流失 |
| `THRESHOLD_GRID` | np.ndarray | `np.arange(0.05, 0.95, 0.05)` | 阈值搜索网格，范围[0.05, 0.95)，步长0.05 |
| `RECALL_TARGET` | float | `0.90` | Recall目标值，用于确定风险分层锚点 |
| `PRECISION_TARGET` | float | `0.60` | Precision目标值，用于确定风险分层锚点 |

---

## 数据预处理变量

### 原始数据加载

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `df` | pd.DataFrame | 从CSV文件加载的原始数据集 |
| `data_path` | Path | 数据文件路径，默认为 `resource/customerchurn.csv` |

### 编码后数据

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `encoded` | pd.DataFrame | 经过编码处理后的数据副本 |
| `pay_dummies` | pd.DataFrame | PaymentMethod字段的独热编码结果（drop_first=True） |
| `InternetService_enc` | pd.Series | InternetService字段经INTERNET_MAP映射后的数值编码 |
| `Contract_enc` | pd.Series | Contract字段经CONTRACT_MAP映射后的数值编码 |
| `Pay_Credit card` | pd.Series | PaymentMethod为"Credit card"的二值哑变量 |
| `Pay_Electronic check` | pd.Series | PaymentMethod为"Electronic check"的二值哑变量 |
| `Pay_Mailed check` | pd.Series | PaymentMethod为"Mailed check"的二值哑变量 |

---

## 模型特征变量

### 原模型特征（7特征）

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `ORIGINAL_FEATURES` / `original_spec.feature_names` | list[str] | `['SeniorCitizen', 'tenure', 'InternetService_enc', 'Contract_enc', 'Pay_Credit card', 'Pay_Electronic check', 'Pay_Mailed check']` |
| `X_orig` | pd.DataFrame | 原模型的完整特征矩阵 |
| `X_train`, `X_test` | pd.DataFrame | 原模型训练集和测试集特征 |

**特征说明：**
- `SeniorCitizen`: 是否为老年人（0/1）
- `tenure`: 用户 tenure（在网时长）
- `InternetService_enc`: 互联网服务类型编码（0=No, 1=DSL, 2=Fiber optic）
- `Contract_enc`: 合同期限编码（0=月付, 1=一年, 2=两年）
- `Pay_Credit card`: 是否使用信用卡支付
- `Pay_Electronic check`: 是否使用电子支票支付
- `Pay_Mailed check`: 是否使用邮寄支票支付

### 重建模型特征（3特征）

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `REBUILT_FEATURES` / `rebuilt_spec.feature_names` | list[str] | `['tenure', 'InternetService_enc', 'Is_Electronic_check']` |
| `X_rebuilt` / `X_r` | pd.DataFrame | 重建模型的完整特征矩阵 |
| `X_train_r`, `X_test_r` | pd.DataFrame | 重建模型训练集和测试集特征 |

**特征说明：**
- `tenure`: 用户 tenure（在网时长）
- `InternetService_enc`: 互联网服务类型编码
- `Is_Electronic_check`: 是否使用电子支票支付（由`Pay_Electronic check`重命名而来）

### 标签变量

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `y` | pd.Series | 目标变量Churn（是否流失） |
| `y_train`, `y_test` | pd.Series | 原模型训练集和测试集标签 |
| `y_train_r`, `y_test_r` | pd.Series | 重建模型训练集和测试集标签 |

---

## 模型训练与评估变量

### 模型对象

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `pipe` / `self._models['original']` | Pipeline | 原模型Pipeline（StandardScaler + LogisticRegression） |
| `pipe_r` / `self._models['rebuilt']` | Pipeline | 重建模型Pipeline（StandardScaler + LogisticRegression） |

### 预测结果

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `y_prob` | np.ndarray | 原模型测试集的流失概率预测值 |
| `y_prob_r` | np.ndarray | 重建模型测试集的流失概率预测值 |
| `y_pred_default` | np.ndarray | 原模型使用默认阈值的二分类预测结果 |
| `y_pred_default_r` | np.ndarray | 重建模型使用默认阈值的二分类预测结果 |

### 评估指标

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `auc_orig` / `metrics['auc']` | float | 原模型测试集AUC值 |
| `auc_rebuilt` / `metrics_r['auc']` | float | 重建模型测试集AUC值 |
| `accuracy` / `metrics['accuracy']` | float | 准确率 |
| `precision` / `metrics['precision']` | float | 流失类的精确率 |
| `recall` / `metrics['recall']` | float | 流失类的召回率 |
| `f1_score` / `metrics['f1']` | float | 流失类的F1分数 |
| `tpr` / `metrics['tpr']` | float | 真正率（True Positive Rate） |
| `fpr` / `metrics['fpr']` | float | 假正率（False Positive Rate） |
| `cm` / `metrics['confusion_matrix']` | list | 混淆矩阵 |
| `tn, fp, fn, tp` | int | 混淆矩阵的四个元素（真负、假正、假负、真正） |

### 交叉验证结果

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `cv` / `cv_r` | StratifiedKFold | 5折分层交叉验证器 |
| `cv_scores` / `cv_scores_r` | dict | 交叉验证各指标得分列表，包含'auc', 'recall', 'precision', 'f1' |

---

## 阈值优化变量

### 阈值遍历计算

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `thresholds` / `THRESHOLD_GRID` | np.ndarray | 阈值搜索数组 [0.05, 0.10, ..., 0.90] |
| `metrics_list` / `metrics_list_r` | list[dict] | 每个阈值对应的precision、recall、f1指标列表 |
| `metrics_df` / `metrics_df_r` | pd.DataFrame | 阈值评估结果的DataFrame形式 |

### 最佳阈值

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `best_idx` / `best_idx_r` | int | F1分数最高时的索引位置 |
| `best_thresh` / `best_thresh_r` | float | 基于F1最优的最佳阈值 |
| `threshold_summary` / `threshold_summary_r` | dict | 包含best_threshold和risk_thresholds的字典 |

### 业务锚点阈值

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `recall_90_thresh` / `anchors['recall_90']` | float | 满足Recall≥90%的最大阈值 |
| `precision_60_thresh` / `anchors['precision_60']` | float | 满足Precision≥60%的最小阈值 |
| `f1_best` / `anchors['f1_best']` | float | F1最优阈值（同best_thresh） |

---

## 风险分层变量

### 风险分层边界

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `anchor_points` / `ordered` | list[float] | 排序后的三个锚点阈值 |
| `low_cut` / `risk_thresholds['low_mid']` | float | 低风险与中等风险的分界点（recall_90锚点） |
| `mid_cut` / `risk_thresholds['high_mid']` | float | 中等风险与较高风险的分界点（f1_best锚点） |
| `high_cut` / `risk_thresholds['high']` | float | 较高风险与高风险的分界点（precision_60锚点） |

### 风险等级

| 风险等级 | 概率范围 | 业务含义 |
|----------|----------|----------|
| 低风险 | P < low_cut | Recall≥90%锚点以下，流失可能性低 |
| 中等风险 | low_cut ≤ P < mid_cut | F1最优锚点区间，需关注 |
| 较高风险 | mid_cut ≤ P < high_cut | Precision≥60%锚点区间，需主动干预 |
| 高风险 | P ≥ high_cut | 高流失风险，需立即采取措施 |

### 风险建议

| 风险等级 | 建议措施 |
|----------|----------|
| 高风险 | 立即联系客户、提供专属优惠、升级服务、客户经理一对一沟通、合同升级优惠 |
| 较高风险 | 主动联系了解体验、针对性产品推荐、满意度调查、限时优惠、定期跟进 |
| 中等风险 | 发送满意度调查、针对性产品推荐、定期跟进、自助服务优化、小幅优惠 |
| 低风险 | 常规维护、产品更新信息、推荐计划、增值服务介绍、保持服务质量 |

---

## 服务类属性

### ChurnModelService类属性

| 属性名 | 类型 | 说明 |
|--------|------|------|
| `self.data_path` | Path | 数据文件路径 |
| `self.original_spec` | ModelSpec | 原模型规格（名称、特征列表、特征描述） |
| `self.rebuilt_spec` | ModelSpec | 重建模型规格（名称、特征列表、特征描述） |
| `self._trained` | bool | 模型是否已训练标志 |
| `self._models` | dict[str, Pipeline] | 存储训练好的模型 {'original': pipe, 'rebuilt': pipe_r} |
| `self._metadata` | dict | 存储模型元数据（规格、阈值、指标、交叉验证结果） |

### ModelSpec数据类属性

| 属性名 | 类型 | 说明 |
|--------|------|------|
| `name` | str | 模型名称（'original' 或 'rebuilt'） |
| `feature_names` | list[str] | 特征名称列表 |
| `feature_description` | str | 特征描述文本 |
| `default_threshold` | float | 默认阈值，默认值为0.35 |

### 验证相关常量

| 变量名 | 类型 | 值 | 说明 |
|--------|------|-----|------|
| `required_columns` | list[str] | `['SeniorCitizen', 'tenure', 'PaymentMethod', 'InternetService', 'Contract']` | 输入数据必需的列 |
| `valid_payment_methods` | list[str] | `['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']` | 有效的支付方式 |
| `valid_internet_services` | list[str] | `['DSL', 'Fiber optic', 'No']` | 有效的互联网服务类型 |
| `valid_contracts` | list[str] | `['Month-to-month', 'One year', 'Two year']` | 有效的合同类型 |

---

## 主要函数说明

### 核心方法

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `_load_raw_data()` | data_path: Path | pd.DataFrame | 读取CSV数据并去重 |
| `_prepare_encodex_frame()` | df: pd.DataFrame | pd.DataFrame | 对数据进行编码处理 |
| `_build_threshold_summary()` | y_true, y_prob | dict | 基于F1和业务锚点计算最佳阈值和风险分层 |
| `_train_models()` | 无 | None | 训练原模型和重建模型 |
| `predict_churn()` | data, model | np.ndarray | 预测客户流失概率 |
| `_get_risk_level()` | probability, model | str | 根据概率获取风险等级 |
| `get_recommendations()` | risk_level | list[str] | 根据风险等级获取建议措施 |
| `process_batch_data()` | df, model | pd.DataFrame | 批量处理数据，添加概率和风险等级列 |
| `get_metadata()` | 无 | dict | 获取模型元数据 |

---

## 数据流概览

```
原始数据 (customerchurn.csv)
    ↓
_load_raw_data() → df (去重后的DataFrame)
    ↓
_prepare_encodex_frame() → encoded (编码后的DataFrame)
    ↓
特征提取 → X_orig (7特征) / X_rebuilt (3特征), y (Churn标签)
    ↓
train_test_split() → 训练集/测试集
    ↓
Pipeline.fit() → 训练模型 (StandardScaler + LogisticRegression)
    ↓
predict_proba() → y_prob (流失概率)
    ↓
_build_threshold_summary() → best_threshold, risk_thresholds
    ↓
_get_risk_level() → 风险等级 (低/中/较高/高)
    ↓
get_recommendations() → 业务建议
```

---

*本文档基于 `model/model_service.py` 和 `resource/churn_logistic.ipynb` 生成，变量已对齐。*
