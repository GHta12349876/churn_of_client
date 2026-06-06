from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_predict,cross_val_score
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
        current_dir = Path(__file__).resolve().parent # __file__表当前文件的绝对路径，Path(__file__)转换为Path对象，.resolve()将路径解析为绝对路径。
        candidates=[
            current_dir / 'customerchurn.csv',
            current_dir.parent / 'customerchurn.csv'
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
            raise FileNotFoundError('Could not locate customerchurn.csv in the app directory or its parent.')


    @staticmethod
    def _load_raw_data(data_path:Path)->pd.DataFrame:
        """读取数据源，返回数据源的二维数组"""
        df=pd.read_csv(data_path)
        return df.drop_duplicates().reset_index(drop=True)  # 为什么这里要删除重复值？不合理


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
    def _build_threshold_summary(y_true:pd.Series,y_prob:np.ndarray)->dict[str,Any]:  # 说实话，这真是个没卵用的东西
        """输出混淆矩阵与阈值，既考虑了业务约束（recall/Precision目标），又提供灵活的阈值选择空间"""
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
            'threshold_grid':THRESHOLD_GRID.tolist(),
            'precisions':precisions,
            'recalls':recalls,
            'f1_score':f1_scores,
            'best_threshold':best_thresh,
            'best_index':best_idx,
            # Top3最佳阈值
            'top_thresholds':[float(THRESHOLD_GRID[index]) for index in np.argsort(f1_scores)[::-1][:3]],
            'anchor_thresholds':anchors,
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
        # 这里的数据处理根本不行啊。异常值检查呢?
        y = encoded['Churn']

        # 接下来一通操作：切分，管道，训练，输出（概率、预测值、混淆矩阵、多维度评分），交叉验证，结果序列化
        X_original = encoded[self.original_spec.feature_names].astype(float).copy()
        X_train_base,X_test_base,y_train_base,y_test_base = train_test_split(X_original,y,test_size=0.3,random_state=42,stratify=y)
        original_pipe = Pipeline([
            ('scaler',StandardScaler()),
            ('clf',LogisticRegression(random_state=42,max_iter=1000,class_weight='balanced'))
        ])
        original_pipe.fit(X_train_base,y_train_base)
        y_pred_proba_base = original_pipe.predict_proba(X_test_base)[:,1]
        original_thresholds = self._build_threshold_summary(y_test_base,y_pred_proba_base)
        original_y_pred_default = (y_pred_proba_base>=DEFAULT_THRESHOLD).astype(int) # 命名为original_y_pred_default是不是更好一些？
        original_default_report = classification_report(y_test_base,original_y_pred_default,target_names=['未流失','流失'],output_dict=True,zero_division=0)
        original_default_cm = confusion_matrix(y_test_base,original_y_pred_default)
        tn_base, fp_base, fn_base, tp_base = original_default_cm.ravel()
        original_default_metrics = {
            'auc':float(roc_auc_score(y_test_base,y_pred_proba_base)),
            'accuracy':float(original_default_report['accuracy']),
            'precision':float(original_default_report['流失']['precision']),  # 嵌套字典访问。当然也可以使用get
            'recall':float(original_default_report['流失']['recall']),
            'f1':float(original_default_report['流失']['f1-score']),
            'tpr':float(tp_base/(tp_base+fn_base) if (tp_base+fn_base)>0 else 0.0), # tpr不就是recall吗？为何多此一举？——也许是为了和fpr对齐
            'fpr':float(fp_base/(fp_base+tn_base) if (fp_base+tn_base)>0 else 0.0),
            'confusion_matrix':original_default_cm.tolist()
            # 为什么总是转换成列表（序列化）？真的有必要吗？——有必要，适用场景：1.API接口返回数据。2.保存到json文件。3.前端展示。 如果不需要的话也可以不转
        }
        cv_base = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
        # 交叉验证，结果序列化，后续可计算均值（平均性能），标准差（置信区间）
        original_cv = {
            'auc':cross_val_score(original_pipe,X_original,y,cv=cv_base,scoring='roc_auc').tolist(),
            'recall':cross_val_score(original_pipe,X_original,y,cv=cv_base,scoring='recall').tolist(),
            'precision':cross_val_score(original_pipe,X_original,y,cv=cv_base,scoring='precision').tolist(),
            'f1':cross_val_score(original_pipe,X_original,y,cv=cv_base,scoring='f1').tolist(),
        }
        # 用于**堆叠集成（Stacking）**时作为下一层模型的输入
        original_oof = cross_val_predict(original_pipe,X_train_base,y_train_base,cv=cv_base,method='predict_proba')[:,1] # 这里为什么又使用训练集

        # 使用3特征再训练
        refit_source = encoded[self.original_spec.feature_names].copy()
        refit_source=refit_source.rename(columns={'Pay_Electronic check':'Is_Electronic_check'})
        X_refit = refit_source[self.rebuilt_spec.feature_names].astype(float).copy()
        X_train_refit, X_test_refit, y_train_refit, y_test_refit = train_test_split(X_refit,y,test_size=0.3,random_state=42,stratify=y)
        refit_pipe = Pipeline([
            ('Scaler',StandardScaler()),
            ('clf',LogisticRegression(random_state=42,max_iter=1000,class_weight='balanced'))
        ])
        refit_pipe.fit(X_train_refit,y_train_refit)
        y_pred_proba_refit = refit_pipe.predict_proba(X_test_refit)[:,1]
        refit_thresholds = self._build_threshold_summary(y_test_refit,y_pred_proba_refit)
        refit_y_pred_default = (y_pred_proba_refit>=DEFAULT_THRESHOLD).astype(int)
        refit_default_report = classification_report(y_test_refit,refit_y_pred_default,target_names=['未流失','流失'],zero_division=0,output_dict=True)
        refit_default_cm = confusion_matrix(y_test_refit,refit_y_pred_default)
        tn_refit, fp_refit, fn_refit, tp_refit = refit_default_cm.ravel()
        refit_default_metrics ={
            'auc':float(roc_auc_score(y_test_refit,y_pred_proba_refit)),
            'accuracy':float(refit_default_report['accuracy']),
            'precision': float(refit_default_report['流失']['precision']),
            'recall': float(refit_default_report['流失']['recall']),
            'f1': float(refit_default_report['流失']['f1-score']),
            'tpr': float(tp_refit / (tp_refit + fn_refit) if (tp_refit + fn_refit) > 0 else 0.0),
            'fpr': float(fp_refit / (fp_refit + tn_refit) if (fp_refit + tn_refit) > 0 else 0.0),
            'confusion_matrix': refit_default_cm.tolist(),
        }
        cv_refit = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        refit_cv = {
            'auc':cross_val_score(refit_pipe,X_refit,y,cv=cv_refit,scoring='roc_auc').tolist(),
            'recall':cross_val_score(refit_pipe,X_refit,y,cv=cv_refit,scoring='recall').tolist(),
            'precision':cross_val_score(refit_pipe,X_refit,y,cv=cv_refit,scoring='precision').tolist(),
            'f1':cross_val_score(refit_pipe,X_refit,y,cv=cv_refit,scoring='f1').tolist()
        }
        refit_oof = cross_val_predict(refit_pipe,X_train_refit,y_train_refit,cv=cv_refit,method='predict_proba')[:,1]
        self._models = {
            'original':original_pipe,
            'refit':refit_pipe
        }
        self._metadata = {
            'original':{
                'spec':self.original_spec,
                'default_threshold':DEFAULT_THRESHOLD,
                'best_threshold':original_thresholds['best_threshold'],
                'risk_thresholds':original_thresholds['risk_thresholds'],
                'metrics':original_default_metrics,
                'cv':original_cv,
                'oof_proba':original_oof.tolist()
            },
            'rebuilt':{
                'spec': self.rebuilt_spec,
                'default_threshold': DEFAULT_THRESHOLD,
                'best_threshold': refit_thresholds['best_threshold'],
                'risk_thresholds': refit_thresholds['risk_thresholds'],
                'metrics': refit_default_metrics,
                'cv': refit_cv,
                'oof_proba': refit_oof.tolist()
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
        """这个和前面的独热编码的特征好像没有对齐，存在代码冗余的问题，到时候看一看"""
        if model == 'original':
            features = pd.DataFrame({
                'SeniorCitizen':df['SeniorCitizen'].astype(float),
                'tenure':df['tenure'].astype(float),
                'InternetService': df['InternetService'].map(INTERNET_MAP).astype(float),
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
            return '中高风险'
        if probability >= thresholds['low_mid']:
            return '中低风险'
        return '低风险'


    def _get_recommendations(self,risk_level:str)->list[str]:
        """给业务的建议"""
        #        recommendations = {
        #     '高风险': [
        #         '立即联系客户，了解不满原因',
        #         '提供专属优惠或折扣方案',
        #         '考虑升级服务或提供增值服务',
        #         '安排客户经理进行一对一沟通',
        #         '提供合同升级优惠（如月付转年付）',
        #     ],
        #     '较高风险': [
        #         '主动联系客户，了解使用体验',
        #         '提供针对性产品推荐或升级方案',
        #         '发送满意度调查并跟进反馈',
        #         '提供限时优惠以提升忠诚度',
        #         '定期跟进客户使用情况',
        #     ],
        #     '中等风险': [
        #         '发送客户满意度调查',
        #         '提供针对性的产品推荐',
        #         '定期跟进客户使用情况',
        #         '提供自助服务优化建议',
        #         '考虑提供小幅优惠以提升忠诚度',
        #     ],
        #     '低风险': [
        #         '保持常规客户关系维护',
        #         '定期发送产品更新信息',
        #         '邀请参与推荐计划',
        #         '提供增值服务介绍',
        #         '保持良好的服务质量',]}
        pass # 自己写


    def process_batch_data(self,df:pd.DataFrame,model:str='original')->pd.DataFrame:
        self._ensure_trained()
        if model not in self._models:
            raise ValueError('model must be "original" or "rebuilt".')
        df_copy = df.copy()
        self._validate_input_frame(df_copy)
        probabilities = self.predict_churn(df_copy,model=model)
        df_copy['ChurnProbability']=probabilities
        df_copy['RiskLevel']=[self._get_risk_level(probability,model=model) for probability in probabilities]
        return df_copy


@lru_cache(maxsize=1)
def get_default_service()->ChurnModelService:
    return ChurnModelService()
