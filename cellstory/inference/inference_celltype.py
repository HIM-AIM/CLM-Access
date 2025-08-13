from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
def generate_atac_celltype_metrics(args,celltypes_labels,predictions):
    
    accuracy = accuracy_score(celltypes_labels, predictions)
    precision = precision_score(celltypes_labels, predictions, average="macro")
    recall = recall_score(celltypes_labels, predictions, average="macro")
    macro_f1 = f1_score(celltypes_labels, predictions, average="macro")
    results = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/macro_f1": macro_f1,
    }
    print(results)
    with open(args.dirpath / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    with open(f"{args.atac_dataset_path}/id2type.pkl", 'rb') as file:
            # 使用pickle.load()函数将读取的内容转换回Python对象
        id2type  = pickle.load(file)
    
    

    celltypes = list(id2type.values())
    
    for i in set([id2type[p] for p in predictions]):
        if i not in celltypes:
            celltypes.remove(i)
    cm = confusion_matrix(celltypes_labels, predictions)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm = pd.DataFrame(cm, index=celltypes[:cm.shape[0]], columns=celltypes[:cm.shape[1]])
    plt.figure(figsize=(len(celltypes)*0.8, len(celltypes)*0.6))  # 动态调整画布大小
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=".1f", 
        cmap="Blues",
        xticklabels=True,
        yticklabels=True,
        annot_kws={'size': 8}  # 调整注释字体大小
    )
    plt.xticks(rotation=45, ha='right')  # 旋转x轴标签
    plt.yticks(rotation=0)
    plt.tight_layout()  # 自动调整布局
    plt.savefig(
        args.dirpath / "confusion_matrix.png", 
        dpi=300,
        bbox_inches='tight'  # 包含所有元素
    )
    plt.close()  # 释放内存

    
