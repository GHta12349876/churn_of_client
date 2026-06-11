from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,precision_score,recall_score,roc_auc_score

INTERNET_MAP={'No':0,"DSL":1,"Fiber optic":2}
CONTRACT_MAP={'Month-to-month':0,'One year':1,'Two year':2}
DEFAULT_THRESHOLD=0.35
THRESHOLD_GRID=np.arange(0.05,0.95,0.05)
RECALL_TARGET=0.90
PRECISION_TARGET=0.60


@dataclass(frozen=True)  # 一个装饰器，自动为类生成__init__,__repr__等方法，frozen=True表示这个类的实例创建后不能修改属性
class ModelSpec:
    name:str
    feature_names:list[str]
    feature_description:str
    default_threshold:float=DEFAULT_THRESHOLD

class ChurnModelService:
    def __init__(self,data_path:str | Path | None=None):
        self.data_path=self._resolve_data_path(data_path)
        self.original_spec=ModelSpec(
            name='original',
            feature_names=[
                'SeniorCitizen',
                'tenure',
                'InternetService_enc',
                'Contract_enc',
                'Pay_Credit card',
                'Pay_Electronic check',
                'Pay_Mailed check'
            ],
            feature_description='SeniorCitizen + tenure + InternetService_enc + Contract_enc + PaymentMethod 3 dummies'
        )
        self.rebuilt_spec=ModelSpec(
            name='rebuilt',
            feature_names=['tenure','InternetService_enc','Is_Electronic_check'],
            feature_description='tenure + InternetService_enc + Is_Electronic_check'
        )
        self._trained=False
        self._models:dict[str,Any]={}
        self._train_models()

    @staticmethod
    def _resolve_data_path(data_path:str | Path |None)-> Path:
        """ 解析数据文件路径"""
        if data_path is not None:
            candidate=Path(data_path)
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f'Could not find dataset at :{candidate}')
        # 数据文件固定在 resource/ 目录下
        return Path(__file__).resolve().parent.parent / 'resource' / 'customerchurn.csv'


    @staticmethod
    def _load_raw_data(data_path:Path)->pd.DataFrame:
        """读取数据源，返回数据源的二维数组"""
        df=pd.read_csv(data_path)
        return df.drop_duplicates().reset_index(drop=True)


    @staticmethod
    def _prepare_encodex_frame(df:pd.DataFrame)->pd.DataFrame:
        """使用独热编码将特征分裂，返回编码后的二维数组"""
        encoded=df.copy()
        encoded['InternetService_enc']=encoded['InternetService'].map(INTERNET_MAP)
        encoded['Contract_enc']=encoded['Contract'].map(CONTRACT_MAP)
        pay_dummies = pd.get_dummies(encoded['PaymentMethod'],prefix='Pay',drop_first=True)
        encoded=pd.concat([encoded,pay_dummies],axis=1)
        return encoded

    @staticmethod
    def _build_threshold_summary(y_true:pd.Series,y_prob:np.ndarray)->dict[str,Any]:
        """基于F1最佳阈值和业务锚点（Recall≥0.9, Precision≥0.6）返回最佳阈值与四档风险分层"""
        precisions:list[float]=[]
        recalls:list[float]=[]
        f1_scores:list[float]=[]

        # 遍历所有阈值计算指标
        for thresh in THRESHOLD_GRID:
            y_pred = (y_prob>=thresh).astype(int)
            precision=precision_score(y_true,y_pred,zero_division=0)
            recall = recall_score(y_true,y_pred,zero_division=0)
            f1=2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
            precisions.append(float(precision))
            recalls.append(float(recall))
            f1_scores.append(float(f1))

        # 找到关键阈值对应的索引
        best_idx = int(np.argmax(f1_scores))
        best_thresh = float(THRESHOLD_GRID[best_idx])

        recall_candidates = np.where(np.array(recalls)>=RECALL_TARGET)[0]  # 注意要使用[0]取出索引元组
        precision_candidates = np.where(np.array(precisions)>=PRECISION_TARGET)[0]
        recall90_idx = int(recall_candidates.max()) if len(recall_candidates) else int(np.argmax(recalls))
        precision60_idx = int(precision_candidates.min()) if len(precision_candidates) else int(np.argmax(precisions))

        # 提取锚点阈值
        anchors = {
            'recall_90' : float(THRESHOLD_GRID[recall90_idx]),
            'f1_best':float(THRESHOLD_GRID[best_idx]),
            'precision_60':float(THRESHOLD_GRID[precision60_idx])
        }

        # 排序得到切割点
        ordered=[value for _,value in sorted(anchors.items(),key=lambda item: item[1])]
        low_cut,mid_cut,high_cut = ordered

        return {
            'best_threshold':best_thresh,
            'risk_thresholds':{
                'low':0.0,
                'low_mid':low_cut,
                'high_mid':mid_cut,
                'high':high_cut
            }
        }


    def _train_models(self)->None:
        df=self._load_raw_data(self.data_path)
        encoded = self._prepare_encodex_frame(df)
        y = encoded['Churn']

        # 原模型：切分，管道，训练，评估，交叉验证
        X_orig = encoded[self.original_spec.feature_names].astype(float).copy()
        X_train,X_test,y_train,y_test = train_test_split(X_orig,y,test_size=0.3,random_state=42,stratify=y)
        pipe = Pipeline([
            ('scaler',StandardScaler()),
            ('clf',LogisticRegression(random_state=42,max_iter=1000,class_weight='balanced'))
        ])
        pipe.fit(X_train,y_train)
        y_prob = pipe.predict_proba(X_test)[:,1]
        threshold_summary = self._build_threshold_summary(y_test,y_prob)
        y_pred_default = (y_prob>=DEFAULT_THRESHOLD).astype(int)
        report = classification_report(y_test,y_pred_default,target_names=['未流失','流失'],output_dict=True,zero_division=0)
        cm = confusion_matrix(y_test,y_pred_default)
        tn, fp, fn, tp = cm.ravel()
        metrics = {
            'auc':float(roc_auc_score(y_test,y_prob)),
            'accuracy':float(report['accuracy']),
            'precision':float(report['流失']['precision']),
            'recall':float(report['流失']['recall']),
            'f1':float(report['流失']['f1-score']),
            'tpr':float(tp/(tp+fn) if (tp+fn)>0 else 0.0),
            'fpr':float(fp/(fp+tn) if (fp+tn)>0 else 0.0),
            'confusion_matrix':cm.tolist()
        }
        cv = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
        cv_scores = {
            'auc':cross_val_score(pipe,X_orig,y,cv=cv,scoring='roc_auc').tolist(),
            'recall':cross_val_score(pipe,X_orig,y,cv=cv,scoring='recall').tolist(),
            'precision':cross_val_score(pipe,X_orig,y,cv=cv,scoring='precision').tolist(),
            'f1':cross_val_score(pipe,X_orig,y,cv=cv,scoring='f1').tolist(),
        }

        # 重建模型：使用3特征再训练
        refit_source = encoded[self.original_spec.feature_names].copy()
        refit_source=refit_source.rename(columns={'Pay_Electronic check':'Is_Electronic_check'})
        X_r = refit_source[self.rebuilt_spec.feature_names].astype(float).copy()
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r,y,test_size=0.3,random_state=42,stratify=y)
        pipe_r = Pipeline([
            ('scaler',StandardScaler()),
            ('clf',LogisticRegression(random_state=42,max_iter=1000,class_weight='balanced'))
        ])
        pipe_r.fit(X_train_r,y_train_r)
        y_prob_r = pipe_r.predict_proba(X_test_r)[:,1]
        threshold_summary_r = self._build_threshold_summary(y_test_r,y_prob_r)
        y_pred_default_r = (y_prob_r>=DEFAULT_THRESHOLD).astype(int)
        report_r = classification_report(y_test_r,y_pred_default_r,target_names=['未流失','流失'],zero_division=0,output_dict=True)
        cm_r = confusion_matrix(y_test_r,y_pred_default_r)
        tn_r, fp_r, fn_r, tp_r = cm_r.ravel()
        metrics_r ={
            'auc':float(roc_auc_score(y_test_r,y_prob_r)),
            'accuracy':float(report_r['accuracy']),
            'precision': float(report_r['流失']['precision']),
            'recall': float(report_r['流失']['recall']),
            'f1': float(report_r['流失']['f1-score']),
            'tpr': float(tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0.0),
            'fpr': float(fp_r / (fp_r + tn_r) if (fp_r + tn_r) > 0 else 0.0),
            'confusion_matrix': cm_r.tolist(),
        }
        cv_r = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        cv_scores_r = {
            'auc':cross_val_score(pipe_r,X_r,y,cv=cv_r,scoring='roc_auc').tolist(),
            'recall':cross_val_score(pipe_r,X_r,y,cv=cv_r,scoring='recall').tolist(),
            'precision':cross_val_score(pipe_r,X_r,y,cv=cv_r,scoring='precision').tolist(),
            'f1':cross_val_score(pipe_r,X_r,y,cv=cv_r,scoring='f1').tolist()
        }
        self._models = {
            'original':pipe,
            'rebuilt':pipe_r
        }
        self._metadata = {
            'original':{
                'spec':self.original_spec,
                'default_threshold':DEFAULT_THRESHOLD,
                'best_threshold':threshold_summary['best_threshold'],
                'risk_thresholds':threshold_summary['risk_thresholds'],
                'metrics':metrics,
                'cv':cv_scores,
            },
            'rebuilt':{
                'spec': self.rebuilt_spec,
                'default_threshold': DEFAULT_THRESHOLD,
                'best_threshold': threshold_summary_r['best_threshold'],
                'risk_thresholds': threshold_summary_r['risk_thresholds'],
                'metrics': metrics_r,
                'cv': cv_scores_r,
            }
        }
        self._trained = True


    def _ensure_trained(self):
        if not self._trained:
            self._train_models()


    def get_metadata(self)->dict[str,Any]:
        self._ensure_trained()
        return {
            'current_model':'original',
            'available_models':['original','rebuilt'],
            'models':{
                model_name:{
                    'feature_count':len(meta['spec'].feature_names),
                    'feature_description':meta['spec'].feature_description,
                    'default_threshold':meta['default_threshold'],
                    'best_threshold':meta['best_threshold'],
                    'risk_thresholds':meta['risk_thresholds'],
                    'metrics':meta['metrics']
                }
                for model_name, meta in self._metadata.items()
            }
        }

    @staticmethod
    def _to_dataframe(data:dict[str,Any]|pd.DataFrame)->pd.DataFrame:
        if isinstance(data,dict):
            return pd.DataFrame([data]).copy()
        if isinstance(data,pd.DataFrame):
            return data.copy()
        raise TypeError('data must be a dict or pandas DataFrame')


    @staticmethod
    def _validate_input_frame(df:pd.DataFrame):
        required_columns = ['SeniorCitizen', 'tenure', 'PaymentMethod', 'InternetService', 'Contract']
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(f'Missing required columns:{','.join(missing_columns)}')
        valid_payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']
        valid_internet_services =['DSL', 'Fiber optic', 'No']
        valid_contracts = ['Month-to-month', 'One year', 'Two year']
        invalid_payments = df[~df['PaymentMethod'].isin(valid_payment_methods)]['PaymentMethod'].unique()
        if len(invalid_payments)>0:
            raise ValueError(f'Invalid PaymentMethod values:{",".join(map(str, invalid_payments))}')
        invalid_internet = df[~df['InternetService'].isin(valid_internet_services)]['InternetService'].unique()
        if len(invalid_internet) > 0:
            raise ValueError(f'Invalid InternetService values: {",".join(map(str,invalid_internet))}')
        invalid_contracts = df[~df['Contract'].isin(valid_contracts)]['Contract'].unique()
        if len(invalid_contracts)>0:
            raise ValueError(f'Invalid Contract values: {",".join(map(str, invalid_contracts))}')


    @staticmethod
    def _build_features(df:pd.DataFrame,model:str)->pd.DataFrame:
        """根据模型类型构建特征DataFrame，用于单条/批量预测；手动编码以保证缺类时列结构固定"""
        if model == 'original':
            features = pd.DataFrame({
                'SeniorCitizen':df['SeniorCitizen'].astype(float),
                'tenure':df['tenure'].astype(float),
                'InternetService_enc': df['InternetService'].map(INTERNET_MAP).astype(float),
                'Contract_enc': df['Contract'].map(CONTRACT_MAP).astype(float),
                'Pay_Credit card': (df['PaymentMethod'] == 'Credit card').astype(float),
                'Pay_Electronic check': (df['PaymentMethod'] == 'Electronic check').astype(float),
                'Pay_Mailed check': (df['PaymentMethod'] == 'Mailed check').astype(float)
            })
            return features
        if model == 'rebuilt':
            features = pd.DataFrame({
                'tenure': df['tenure'].astype(float),
                'InternetService_enc': df['InternetService'].map(INTERNET_MAP).astype(float),
                'Is_Electronic_check': (df['PaymentMethod'] == 'Electronic check').astype(float)
            })
            return features
        raise ValueError('model must be "original" or "rebuilt".')


    def predict_churn(self,data:dict[str,Any]|pd.DataFrame,model:str='original')->np.ndarray:
        self._ensure_trained()
        if model not in self._models:
            raise ValueError('model must be "original" or "rebuilt".')
        df= self._to_dataframe(data)
        self._validate_input_frame(df)
        features = self._build_features(df,model=model)
        model_obj = self._models[model]
        probabilities = model_obj.predict_proba(features)[:,1]
        return probabilities


    def _get_risk_level(self,probability:float,model:str='original')->str:
        self._ensure_trained()
        if model not in self._metadata:
            raise ValueError('model must be "original" or "rebuilt".')
        thresholds = self._metadata[model]['risk_thresholds']
        if probability >= thresholds['high']:
            return '高风险'
        if probability >= thresholds['high_mid']:
            return '较高风险'
        if probability >= thresholds['low_mid']:
            return '中等风险'
        return '低风险'


    def get_recommendations(self,risk_level:str)->list[str]:
        """给业务的建议"""
        recommendations = {
            '高风险': [
                '立即联系客户，了解不满原因',
                '提供专属优惠或折扣方案',
                '考虑升级服务或提供增值服务',
                '安排客户经理进行一对一沟通',
                '提供合同升级优惠（如月付转年付）',
            ],
            '较高风险': [
                '主动联系客户，了解使用体验',
                '提供针对性产品推荐或升级方案',
                '发送满意度调查并跟进反馈',
                '提供限时优惠以提升忠诚度',
                '定期跟进客户使用情况',
            ],
            '中等风险': [
                '发送客户满意度调查',
                '提供针对性的产品推荐',
                '定期跟进客户使用情况',
                '提供自助服务优化建议',
                '考虑提供小幅优惠以提升忠诚度',
            ],
            '低风险': [
                '保持常规客户关系维护',
                '定期发送产品更新信息',
                '邀请参与推荐计划',
                '提供增值服务介绍',
                '保持良好的服务质量',
            ]
        }
        return recommendations.get(risk_level, [])


    def process_batch_data(self,df:pd.DataFrame,model:str='original')->pd.DataFrame:  # 处理批量数据
        self._ensure_trained()
        if model not in self._models:
            raise ValueError('model must be "original" or "rebuilt".')
        df_copy = df.copy()
        self._validate_input_frame(df_copy)
        probabilities = self.predict_churn(df_copy,model=model)
        df_copy['ChurnProbability'] = probabilities
        df_copy['RiskLevel']=[self._get_risk_level(probability,model=model) for probability in probabilities]
        return df_copy


@lru_cache(maxsize=1)
def get_default_service()->ChurnModelService:
    return ChurnModelService()
