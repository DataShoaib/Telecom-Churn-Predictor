import pandas as pd, numpy as np, os, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from logger.logger import get_logger
from src.feature.feature_eng import column_preprocessor

logger=get_logger('Model-Building')

def load_data(file_path:str)->pd.DataFrame:
    try: logger.info(f'loading data from {file_path}'); return pd.read_csv(file_path)
    except Exception as e: logger.error(f'error loading data: {e}'); raise

def encode_target(series): return (series=='Yes').astype(int)

def splitting_data(data:pd.DataFrame):
    try:
        logger.info('starting data splitting')
        X=data.drop('churn',axis=1); y=encode_target(data['churn'])
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        logger.info('data splitting completed successfully')
        return X_train,X_test,y_train,y_test
    except Exception as e: logger.error(f'error while splitting data: {e}'); raise

def model_building(X_train:pd.DataFrame,y_train:pd.Series)->Pipeline:
    try:
        logger.info('starting model building')
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        model=LogisticRegression(max_iter=2000,tol=0.01,class_weight='balanced',C=0.3359818286283781,solver='lbfgs')
        model_pipe=Pipeline([('preprocessor',column_preprocessor()),('model',model)])
        cv_score=cross_val_score(model_pipe,X_train,y_train,cv=skf,scoring='recall')
        model_pipe.fit(X_train,y_train)
        logger.info(f'model built successfully | cv recall: {cv_score.mean()}')
        return model_pipe
    except Exception as e: logger.error(f'error while building model: {e}'); raise

def save_model(model:Pipeline,save_model_path:str)->None:
    try:
        logger.info(f'saving model at {save_model_path}')
        os.makedirs(save_model_path,exist_ok=True)
        with open(os.path.join(save_model_path,'model.pkl'),'wb') as f: pickle.dump(model,f)
        logger.info('model saved successfully')
    except Exception as e: logger.error(f'error while saving model: {e}'); raise

def save_split_data(X_train:pd.DataFrame,X_test:pd.DataFrame,y_train:pd.Series,y_test:pd.Series,save_data_path:str)->None:
    try:
        logger.info(f'saving split data at {save_data_path}')
        path=os.path.join(save_data_path,'split'); os.makedirs(path,exist_ok=True)
        X_train.to_csv(os.path.join(path,'x_train.csv'),index=False) 
        X_test.to_csv(os.path.join(path,'x_test.csv'),index=False)
        y_train.to_csv(os.path.join(path,'y_train.csv'),index=False)
        y_test.to_csv(os.path.join(path,'y_test.csv'),index=False)
        logger.info('split data saved successfully')
    except Exception as e: logger.error(f'error saving split data: {e}'); raise

def main():
    data=load_data('data/processed/feature_eng_data.csv')
    X_train,X_test,y_train,y_test=splitting_data(data)
    model=model_building(X_train,y_train)
    save_model(model,'models')
    save_split_data(X_train,X_test,y_train,y_test,'data/processed')

if __name__=="__main__": main()