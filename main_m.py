import numpy as np
import argparse 
import torch

from train_m import TrainerDeepSVDD
from preprocess import get_mnist
from test import eval
import pandas as pd

from collections import Counter
from sklearn.metrics import roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random

def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

if __name__ == '__main__':

    set_seed()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=150, help="number of epochs")
    parser.add_argument("--num_epochs_ae", type=int, default=150, help="number of epochs for the pretraining")
    parser.add_argument("--patience", type=int, default=50, help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.5e-6, help='Weight decay hyperparameter for the L2 regularization')
    parser.add_argument('--weight_decay_ae', type=float, default=0.5e-3, help='Weight decay hyperparameter for the L2 regularization')
    parser.add_argument('--lr_ae', type=float, default=1e-4, help='learning rate for autoencoder')
    parser.add_argument('--lr_milestones', type=list, default=[50], help='Milestones at which the scheduler multiply the lr by 0.1')
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size")
    parser.add_argument('--pretrain', type=bool, default=True, help='Pretrain the network using an autoencoder')
    parser.add_argument('--latent_dim', type=int, default=32, help='Dimension of the latent variable z')
    
    # 2 classes
    parser.add_argument("--normal_class", type=int, default=0, help="normal class")
    parser.add_argument("--abnormal_class", type=int, default=1, help="abnormal_class")
    # dataset
    parser.add_argument("--dataset", type=str, default="mnist", help="type of dataset")
    parser.add_argument("--num_of_classes", type=int, default=10, help="# of class")

    #parsing arguments.
    args = parser.parse_args() 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for _ in range(args.num_of_classes):

        # Write additional details to the results file
        result_file_path = f'/home/cal-05/hj/SVDD/result/{args.dataset}/{args.normal_class}/auc_scores.txt'
        with open(result_file_path, 'w') as file:
            file.write(f'Normal Class / Abnormal Class : ROC AUC Score\n')

        # 하나의 normal class에 대해 여러개의 abormal class    
        for abnormal in range(args.num_of_classes):
            if abnormal == args.normal_class:
                continue
            
            args.abnormal_class = abnormal
            print(args.abnormal_class)

            data = get_mnist(args)
            deep_SVDD = TrainerDeepSVDD(args, data, device)

            if args.pretrain:
                deep_SVDD.pretrain()
            deep_SVDD.train()

            '''
                test and visualization
            '''

            indices, labels, scores = eval(deep_SVDD.net, deep_SVDD.c, data[1], device)
            print(labels)

            # rou_auc_score
            roc_auc = roc_auc_score(labels, scores) * 100

            # confusion matrix
            normal_scores = [score for label, score in zip(labels, scores) if label == 0]
            normal_max_dist = max(normal_scores)
            abnormal_scores = [score for label, score in zip(labels, scores) if label == 1]
            abnormal_max_dist = min(abnormal_scores)
            
            threshold = sum(normal_scores) / len(normal_scores)
            predictions = [1 if score >= threshold else 0 for score in scores]
            cf_matrix = confusion_matrix(labels, predictions)
            
            group_names = ["TN", "FP", "FN", "TP"]
            group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
            group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
            cf_labels = ["{0}\n{1}\n({2})".format(v1, v2, v3) for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
            cf_labels = np.asarray(cf_labels).reshape(2,2)
            
            sns.heatmap(cf_matrix, annot=cf_labels, fmt='', cmap='Blues', xticklabels=['Predicted Normal', 'Predicted Abnormal'], yticklabels=['Actual Normal', 'Actual Abnormal'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix for Class {args.normal_class} / {args.abnormal_class}')
            plt.savefig(f'/home/cal-05/hj/SVDD/result/{args.dataset}/{args.normal_class}/Confusion_Matrix_{args.abnormal_class}.png')
            plt.close()

            # 가장 이상적인 normal, abnormal class 5개 시각화 
            normal_indices_scores = [(idx, score) for idx, score, label in zip(indices, scores, labels) if label == 0]
            abnormal_indices_scores = [(idx, score) for idx, score, label in zip(indices, scores, labels) if label == 1]

            normal_indices_scores.sort(key=lambda x: x[1])  
            abnormal_indices_scores.sort(key=lambda x: x[1], reverse=True)  
            top_normal_images_scores = normal_indices_scores[:5]
            top_abnormal_images_scores = abnormal_indices_scores[:5]

            fig, axs = plt.subplots(2, 5, figsize=(15, 6))
            for i, (idx, score) in enumerate(top_normal_images_scores):
                img = data[1].dataset[idx][0]  
                axs[0, i].imshow(img.squeeze(), cmap='gray')
                axs[0, i].set_title(f'Normal\nScore: {score:.2f}')
                axs[0, i].axis('off')

            for i, (idx, score) in enumerate(top_abnormal_images_scores):
                img = data[1].dataset[idx][0]  
                axs[1, i].imshow(img.squeeze(), cmap='gray')
                axs[1, i].set_title(f'Abnormal\nScore: {score:.2f}')
                axs[1, i].axis('off')

            plt.tight_layout()
            plt.savefig(f'/home/cal-05/hj/SVDD/result/{args.dataset}/{args.normal_class}/Class_{args.abnormal_class}_visualization.png')
            plt.close(fig)

            print(f'Finished processing class {args.normal_class} / {args.abnormal_class}.')

            normal_scores_np = np.array(normal_scores)
            abnormal_scores_np = np.array(abnormal_scores)     

            with open(result_file_path, 'a') as file:
                file.write(f' {args.normal_class}    /    {args.abnormal_class} : {roc_auc:.2f}%\n')
                file.write(f'Label counts: {Counter(labels)}\n')
                file.write(f'Prediction counts: {Counter(predictions)}\n')
                file.write(f'Normal Scores - Min: {np.min(normal_scores_np):.2f}, Max: {np.max(normal_scores_np):.2f}, Mean: {np.mean(normal_scores_np):.2f}\n')
                file.write(f'Abnormal Scores - Min: {np.min(abnormal_scores_np):.2f}, Max: {np.max(abnormal_scores_np):.2f}, Mean: {np.mean(abnormal_scores_np):.2f}\n')

        args.normal_class += 1