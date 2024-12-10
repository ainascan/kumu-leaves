import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import argparse
import yaml
import time
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
from sklearn.utils import resample
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, f1_score, precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import VotingClassifier
from PIL import Image

sns.set_theme(style='whitegrid')
logging.basicConfig(level=logging.INFO)


def plot_classifier(model, x_test, y_test, y_pred):
    f1 = f1_score(y_test, y_pred)

    # plot precision recall curve
    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(x_test)[:, 1])
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
    conf_matrix = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    
    # padding at top
    plt.subplots_adjust(top=0.85)
    
    ax[0].set_title('Confusion Matrix')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax[0])

    ax[1].plot(recall, precision, color='b')
    ax[1].fill_between(recall, precision, alpha=0.2, color='b')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title('Precision Recall Curve')

    ax[2].plot(fpr, tpr, color='b')
    ax[2].fill_between(fpr, tpr, alpha=0.2, color='b')
    ax[2].set_xlabel('False Positive Rate')
    ax[2].set_ylabel('True Positive Rate')
    ax[2].set_title('ROC Curve')

    model_name = model.__class__.__name__
    plt.suptitle(f'Accuracy of {model_name} | F1 {f1:.2f}')

    os.makedirs(os.path.join('output', model_name.lower()), exist_ok=True)
    plt.savefig(os.path.join('output', model_name.lower(), 'metrics.png'))


def test_model(search, x_train, x_test, y_train, y_test, grid=True):
    search.fit(x_train, y_train)
    
    model = search
    params = {}
    if grid:
        model = search.best_estimator_
        params = search.best_params_
    
    y_pred = model.predict(x_test)
    
    plot_classifier(model, x_test, y_test, y_pred)
    
    f1 = f1_score(y_test, y_pred)
    
    mlflow.log_metric('f1', f1)
    
    for key, value in params.items():
        mlflow.log_param(key, value)
        
    mlflow.log_artifacts(os.path.join('output', model.__class__.__name__.lower()))

    signature = infer_signature(x_train, model.predict(x_train))
    
    model_name = f'kina_{model.__class__.__name__.lower()}'
    mlflow.sklearn.log_model(model, model_name, signature=signature)
    
    return model, params


def get_dataset(dir, seed):
    classes = ['healthy', 'defective']
    
    df = []
    for i, cls in enumerate(classes):     
        for file in os.listdir(os.path.join(dir, cls)):
            df.append((os.path.join(dir, cls, file), i))
            
    df = pd.DataFrame(df, columns=['image_path', 'defective'])
    
    df_majority = df[df['defective'] == 0]
    df_minority = df[df['defective'] == 1]
    
    if len(df_majority) < len(df_minority):
        df_majority, df_minority = df_minority, df_majority
        
    df_majority_downsampled = resample(df_majority,
                                        replace=False,
                                        n_samples=len(df_minority),
                                        random_state=seed)

    # combine majority class with upsampled minority class
    df = pd.concat([df_majority_downsampled, df_minority])

    # set column 'sett' to be training, validation, or test
    df['sett'] = np.random.choice(['train', 'val', 'test'], size=len(df), p=[0.7, 0.1, 0.2])

    return df


def get_dominant_colors(image_path, amount=64):
    image = Image.open(image_path)
    paletted = image.convert('P', palette=Image.ADAPTIVE, colors=amount)
    palette = paletted.getpalette()
    color_idxs = paletted.getcolors()
    colors = np.array([palette[idx*3:idx*3+3] for _, idx in color_idxs]) / 255
    colors = colors[np.argsort(np.linalg.norm(colors, axis=1))]
    colors = colors.flatten().tolist()
    colors += [0] * ((amount*3) - len(colors))
    return colors


with __name__ == "__main__":
    seed = 12348128
    mlflow_uri = 'http://10.20.0.106:8081'
    run_name = time.strftime('%Y-%m-%d %H:%M:%S')

    df = get_dataset('/home/jack/Documents/Workspace/ainascan/kumu/data/kina/patches', seed)    

    df['colors'] = df['image_path'].apply(get_dominant_colors)
    
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment('Kina Leaf Classification')
    
    mlflow.log_param('seed', seed)
    
    x = np.array(df['colors'].tolist())
    y = df['defective'].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    
    mlflow.start_run(run_name=f"bayes_{run_name}")
    bayes = test_model(
        GridSearchCV(
            MultinomialNB(),
            param_grid={'alpha': [0.001, 0.01, 0.1, 1, 10, 50, 100]},
            n_jobs=-1,
            cv=5,
            scoring='f1_weighted',
        ),
        x_train, x_test, y_train, y_test
    )
    mlflow.end_run()

    mlflow.start_run(run_name=f"xgb_{run_name}")
    xgb = test_model(
        GridSearchCV(
            XGBClassifier(),
            param_grid={'n_estimators': [200,230,250,260], 'max_depth': [30, 45, 50, 100]},
            n_jobs=-1,
            cv=5,
            scoring='f1_weighted',
        ),
        x_train, x_test, y_train, y_test
    )
    mlflow.end_run()

    mlflow.start_run(run_name=f"voting_{run_name}")
    voting = test_model(
        VotingClassifier(
            estimators=[
                ('bayes', bayes),
                ('xgb', xgb)
            ],
            voting='soft',
            n_jobs=-1
        ),
        x_train, x_test, y_train, y_test,
        grid=False
    )
    mlflow.end_run()
