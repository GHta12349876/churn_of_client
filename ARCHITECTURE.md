系统架构与组件关系图

## 目录
- [整体架构图](#整体架构图)
- [数据流图](#数据流图)
- [API路由映射](#api路由映射)

---

## 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端展示层 (Frontend)                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    templates/index.html                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │   │
│  │  │  单个预测Tab  │  │  批量预测Tab  │  │  模型切换组件   │  │   │
│  │  │              │  │              │  │                │  │   │
│  │  │ - 表单输入    │  │ - 文件上传    │  │ - 下拉选择器    │  │   │
│  │  │ - 概率展示    │  │ - 统计卡片    │  │ - 阈值提示      │  │   │
│  │  │ - 风险等级    │  │ - 结果表格    │  │                │  │   │
│  │  │ - 策略建议    │  │ - 导出Excel  │  │                │  │   │
│  │  └──────────────┘  └──────────────┘  └────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP Requests (Fetch API)
                             │ AJAX / RESTful API
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       后端服务层 (Backend)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                        app.py (Flask)                     │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │                  路由控制器 (Routes)                 │  │   │
│  │  │                                                      │  │   │
│  │  │  GET  /            → index()         渲染首页        │  │   │
│  │  │  POST /predict     → predict()       单条预测        │  │   │
│  │  │  POST /batch_predict→ batch_predict() 批量预测       │  │   │
│  │  │  POST /export      → export_results() 导出结果       │  │   │
│  │  │  GET/POST /model   → model_config()  模型配置        │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │                           │                                │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │               业务逻辑封装 (Wrappers)                │  │   │
│  │  │                                                      │  │   │
│  │  │  predict_churn(data, model)                          │  │   │
│  │  │  get_risk_level(probability, model)                  │  │   │
│  │  │  get_recommendations(risk_level)                     │  │   │
│  │  │  process_batch_data(df, model)                       │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │                           │                                │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │              全局服务实例 (Singleton)                 │  │   │
│  │  │                                                      │  │   │
│  │  │  prediction_service = get_default_service()          │  │   │
│  │  │  CURRENT_MODEL = 'original'                          │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ Python Method Calls
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       模型服务层 (Model Service)                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              model/model_service.py                       │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │              ChurnModelService 类                   │  │   │
│  │  │                                                      │  │   │
│  │  │  核心方法:                                            │  │   │
│  │  │  ├─ _train_models()          训练双模型              │  │   │
│  │  │  ├─ predict_churn()          预测流失概率            │  │   │
│  │  │  ├─ _get_risk_level()        获取风险等级            │  │   │
│  │  │  ├─ get_recommendations()    获取业务建议            │  │   │
│  │  │  ├─ process_batch_data()     批量处理                │  │   │
│  │  │  ├─ get_metadata()           获取模型元数据          │  │   │
│  │  │  └─ _build_features()        构建特征矩阵            │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │                           │                                │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │              ModelSpec 数据类                       │  │   │
│  │  │                                                      │  │   │
│  │  │  ├─ original_spec  (7特征)                          │  │   │
│  │  │  └─ rebuilt_spec   (3特征)                          │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │                           │                                │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │              全局常量配置                            │  │   │
│  │  │                                                      │  │   │
│  │  │  INTERNET_MAP, CONTRACT_MAP                         │  │   │
│  │  │  DEFAULT_THRESHOLD, THRESHOLD_GRID                  │  │   │
│  │  │  RECALL_TARGET, PRECISION_TARGET                    │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ sklearn/pandas/numpy
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      数据与依赖层 (Data & ML)                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  resource/       │  │  scikit-learn    │  │  pandas/     │  │
│  │  customerchurn.  │  │  - LogisticReg.  │  │  numpy       │  │
│  │  csv             │  │  - Pipeline      │  │              │  │
│  │                  │  │  - StandardScaler│  │              │  │
│  │                  │  │  - Metrics       │  │              │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```
---

---

## 数据流图

### 单条预测数据流

```
┌──────────────┐
│  用户输入表单  │
│              │
│ SeniorCitizen│ ──┐
│ tenure       │   │
│ PaymentMethod│   │
│ InternetSvc  │   │
│ Contract     │   │
└──────────────┘   │
                   │
                   ▼
┌──────────────────────────────────────────┐
│  app.py: predict()                       │
│  ┌────────────────────────────────────┐  │
│  │ 构建data字典                         │  │
│  │ {                                  │  │
│  │   'SeniorCitizen': int,            │  │
│  │   'tenure': int,                   │  │
│  │   'PaymentMethod': str,            │  │
│  │   'InternetService': str,          │  │
│  │   'Contract': str                  │  │
│  │ }                                  │  │
│  └────────────────────────────────────┘  │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  model_service.py: predict_churn()       │
│                                          │
│  1. _to_dataframe() → DataFrame([data])  │
│  2. _validate_input_frame()  验证合法性   │
│  3. _build_features()       特征工程     │
│     ┌─────────────────────────────────┐  │
│     │ original模型 (7特征):           │  │
│     │ - SeniorCitizen (float)         │  │
│     │ - tenure (float)                │  │
│     │ - InternetService_enc (0/1/2)   │  │
│     │ - Contract_enc (0/1/2)          │  │
│     │ - Pay_Credit card (0/1)         │  │
│     │ - Pay_Electronic check (0/1)    │  │
│     │ - Pay_Mailed check (0/1)        │  │
│     └─────────────────────────────────┘  │
│     ┌─────────────────────────────────┐  │
│     │ rebuilt模型 (3特征):            │  │
│     │ - tenure (float)                │  │
│     │ - InternetService_enc (0/1/2)   │  │
│     │ - Is_Electronic_check (0/1)     │  │
│     └─────────────────────────────────┘  │
│  4. model.predict_proba(features)        │
│     → probabilities [0.0 ~ 1.0]          │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  app.py: 接收概率值                       │
│  probability = float(predict_churn()[0]) │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  model_service.py: _get_risk_level()     │
│                                          │
│  根据risk_thresholds分层:                 │
│  - low (0.0)                             │
│  - low_mid (recall_90锚点)               │
│  - high_mid (f1_best锚点)                │
│  - high (precision_60锚点)               │
│                                          │
│  返回: "低风险" | "中等风险"              │
│        "较高风险" | "高风险"              │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  model_service.py: get_recommendations() │
│                                          │
│  根据风险等级返回建议列表:                  │
│  - 高风险: 5条紧急措施                    │
│  - 较高风险: 5条主动干预措施              │
│  - 中等风险: 5条关注措施                  │
│  - 低风险: 5条常规维护措施                │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  app.py: 组装JSON响应                     │
│  {                                       │
│    'success': true,                      │
│    'probability': 85.23,                 │
│    'risk_level': '高风险',                │
│    'recommendations': [...],             │
│    'model_used': 'original'              │
│  }                                       │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────┐
│  前端展示结果  │
│              │
│ - 流失概率85% │
│ - 高风险标签  │
│ - 5条建议    │
└──────────────┘
```

### 批量预测数据流

```
┌──────────────┐
│ 用户上传文件  │
│ CSV/Excel    │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  app.py: batch_predict()                 │
│  ┌────────────────────────────────────┐  │
│  │ 1. 接收file                        │  │
│  │ 2. 判断格式(.csv/.xlsx/.xls)       │  │
│  │ 3. pd.read_csv() / read_excel()   │  │
│  │ 4. 得到DataFrame df                │  │
│  └────────────────────────────────────┘  │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  model_service.py: process_batch_data()  │
│                                          │
│  1. df.copy() → df_copy                  │
│  2. _validate_input_frame(df_copy)       │
│  3. predict_churn(df_copy) → probabilities│
│  4. df_copy['ChurnProbability'] = probs  │
│  5. for prob in probs:                   │
│       df_copy['RiskLevel'] =             │
│         _get_risk_level(prob)            │
│  6. 返回增强后的DataFrame                 │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  app.py: 组装结果数组                     │
│  ┌────────────────────────────────────┐  │
│  │ for row in result_df:              │  │
│  │   results.append({                 │  │
│  │     'row': index+1,                │  │
│  │     'monthly_charges': ...,        │  │
│  │     'senior_citizen': ...,         │  │
│  │     'tenure': ...,                 │  │
│  │     'payment_method': ...,         │  │
│  │     'internet_service': ...,       │  │
│  │     'contract': ...,               │  │
│  │     'probability': ...,            │  │
│  │     'risk_level': ...              │  │
│  │   })                               │  │
│  │                                    │  │
│  │ app.config['LAST_BATCH_RESULTS'] = │  │
│  │   result_df  # 保存用于导出         │  │
│  └────────────────────────────────────┘  │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  返回JSON响应                             │
│  {                                       │
│    'success': true,                      │
│    'count': 100,                         │
│    'results': [...],                     │
│    'model_used': 'original'              │
│  }                                       │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  前端(index.html)                        │
│  ┌────────────────────────────────────┐  │
│  │ 1. 计算各风险等级数量               │  │
│  │ 2. 渲染统计卡片(总数/高/较高/中/低) │  │
│  │ 3. 渲染结果表格                     │  │
│  │ 4. 显示成功消息                     │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

---

## API路由映射

### Flask路由与处理函数对照表

| HTTP方法 | 路由路径 | 处理函数 | 功能说明 | 请求参数 | 响应格式 |
|---------|---------|---------|---------|---------|---------|
| GET | `/` | `index()` | 渲染首页HTML | 无 | HTML页面 |
| POST | `/predict` | `predict()` | 单条客户流失预测 | Form: senior_citizen, tenure, payment_method, internet_service, contract | JSON: {success, probability, risk_level, recommendations, model_used} |
| POST | `/batch_predict` | `batch_predict()` | 批量客户流失预测 | File: CSV/Excel文件 | JSON: {success, count, results[], model_used} |
| POST | `/export` | `export_results()` | 导出预测结果为Excel | 无（从app.config读取） | Excel文件下载 |
| GET | `/model` | `model_config()` | 获取模型元数据 | 无 | JSON: {current_model, available_models, models:{...}} |
| POST | `/model` | `model_config()` | 切换当前使用的模型 | JSON: {model: 'original'\|'rebuilt'} | JSON: {success, current_model, message} |

---

## 组件依赖关系总结

```
┌─────────────────────────────────────────────────────────────┐
│                     依赖层级关系                              │
└─────────────────────────────────────────────────────────────┘

Layer 4: 前端展示层 (templates/index.html)
  ├─ HTML/CSS/JavaScript
  ├─ Fetch API (HTTP通信)
  └─ 依赖 Layer 3 提供的REST API

Layer 3: Web服务层 (app.py)
  ├─ Flask框架
  ├─ 路由控制
  ├─ 请求/响应处理
  ├─ 文件上传/下载
  └─ 依赖 Layer 2 提供的业务逻辑

Layer 2: 业务逻辑层 (model/model_service.py)
  ├─ ChurnModelService类
  ├─ 数据预处理
  ├─ 特征工程
  ├─ 模型推理
  ├─ 风险评估
  └─ 依赖 Layer 1 的ML库和数据文件

Layer 1: 基础设施层
  ├─ scikit-learn (LogisticRegression, Pipeline, Metrics)
  ├─ pandas (数据处理)
  ├─ numpy (数值计算)
  └─ resource/customerchurn.csv (数据源)
```

---

## 关键设计模式

### 1. 单例模式 (Singleton)
```python
@lru_cache(maxsize=1)
def get_default_service() -> ChurnModelService:
    return ChurnModelService()
```
- 确保全局只有一个`ChurnModelService`实例
- 避免重复训练模型，提升性能
- 在`app.py`启动时调用一次，后续复用

### 2. 管道模式 (Pipeline)
```python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(...))
])
```
- 标准化和分类器封装为一个单元
- 保证训练和预测时的预处理一致性
- 简化部署流程

### 3. 工厂模式 (Factory)
```python
_build_features(df, model='original')  # 根据model参数构建不同特征
```
- 根据模型类型动态生成特征矩阵
- 支持原模型(7特征)和重建模型(3特征)

### 4. 策略模式 (Strategy)
```python
risk_thresholds = {
    'low': 0.0,
    'low_mid': recall_90_anchor,
    'high_mid': f1_best_anchor,
    'high': precision_60_anchor
}
```
- 基于不同业务锚点定义风险分层策略
- 可灵活调整阈值而不改变代码结构


