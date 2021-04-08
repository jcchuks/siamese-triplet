import os

import torch
import numpy as np

from sklearn import svm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# import umap
from constants import *


#os.makedirs(result_folder, exist_ok=True)
step = 0

umap_model = None

sns.set(style='white', context='poster', rc={'figure.figsize': (14, 10)})


all_data = [[], []]


colormap = plt.cm.Set1 #nipy_spectral, Set1,Paired  
colorst = [colormap(i) for i in np.linspace(0, 0.9, shared_params["classes"])]


def get_cmap(n, name='tab10'):
    color = plt.cm.get_cmap(name, n)
    return color #.reshape(1,-1)
    
def draw_umap(embeddings, n_neighbors=15,
              min_dist=0.1, n_components=2,
              metric='euclidean', title=''):
    global umap_model, result_folder, step, folder, colorst 

    fig = plt.figure()

    _folder = cluster_folder + folder
    
    os.makedirs(_folder, exist_ok=True)

    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)


    for idx,u1 in enumerate(embeddings):
        if not u1: continue
        lu1  = len(u1)
        u1 = torch.stack(u1)
        if n_components == 1: 
            ax.scatter(u1[:, 0], range(len(u1)))
        if n_components == 2: 
            ax.scatter(u1[:, 0], u1[:, 1], label=str(idx)) 
        if n_components == 3: 
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_zticks([])
            ax.set_zticklabels([]) 
            ax.scatter(u1[:, 0], u1[:, 1], u1[:, 2], label=str(idx)) 
    for idx, axx in enumerate(ax.collections):
        axx.set_color(colorst[idx])
    ax.legend(fontsize = "small")
    plt.title(title, fontsize=18)
    plt.savefig(os.path.join(_folder, "pic"+str(step).rjust(4,"0")), bbox_inches='tight')
    #plt.show()
    step += 1
    plt.close()




def fit(test_name, train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
        metrics=[],
        start_epoch=0, pegged_metric=None, best_initial_loss=float("inf"), save_path="./"):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    best_train_nzt = float('inf')
    best_test_nzt = float("inf")
    training_loss = np.array([])
    validation_loss = np.array([])
    training_metric = np.array([])
    validation_metric = np.array([])
    best_training_loss = best_validation_loss = best_initial_loss
    # for epoch in range(0, start_epoch):
    #     scheduler.step()
    global step 
    step = 0
    for epoch in range(start_epoch, n_epochs):
        train_nzt = test_nzt = 0
        current_training_loss = current_validation_loss = 0
        svm_model = svm.SVC(kernel='rbf')
        # Train stage
        train_loss, metrics, anchors = train_epoch(svm_model, train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            if pegged_metric and pegged_metric == metric.metric_name:
                current_training_loss = train_loss
                train_nzt = metric.value()
                training_loss = np.append(training_loss, current_training_loss)
                training_metric = np.append(training_metric, metric.value())
            message += '\t{}: {}'.format(metric.name(), metric.value()) 
           
        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            if pegged_metric and pegged_metric == metric.metric_name:
                current_validation_loss = val_loss
                test_nzt = metric.value()
                validation_loss = np.append(validation_loss, current_validation_loss)
                validation_metric = np.append(validation_metric, metric.value())
            message += '\t{}: {}'.format(metric.name(), metric.value())

        if test_nzt <= best_test_nzt and train_nzt <= best_train_nzt:  
            #current_validation_loss <= best_validation_loss and current_training_loss < best_training_loss:
            #best_training_loss = current_training_loss
            #best_validation_loss = current_validation_loss
            best_train_nzt = train_nzt
            best_test_nzt = test_nzt

            os.makedirs(save_path, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validation_loss': current_validation_loss,
                'training_loss': current_training_loss,
                'anchors': anchors,
                'svm': svm_model,
                'message': message
            }, os.path.join(save_path, test_name + best_model_file_name))

            with open(os.path.join(result_folder, validation_file), 'w') as file:
                file.write(message)
        print(message)
        scheduler.step()
    return training_loss, validation_loss, training_metric, validation_metric


def train_epoch(svm, train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, trainName="trainVideo"):
    global folder
    folder = trainName 
    
    for metric in metrics:
        metric.reset()
    model.train()
    losses = []
    total_loss = 0
    all_data = [[] for _ in range(shared_params["classes"])]
    anchors = [ None for _ in range(shared_params["classes"])]
    batch_count = len(train_loader) or 1
    
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()
        outputs = model(*data)
        #print(outputs.shape, target.shape)
        
        outs, tars = outputs.cpu().detach().numpy(), target.cpu().detach().numpy()
        svm.fit(outs, tars) 
        
        if type(outputs) not in (tuple, list):
            outputs = (outputs,)
        
        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for embedding, label in zip(loss_inputs[0], loss_inputs[1]):
            if cuda:  
              embedding_copy = embedding.cpu().clone().detach()
              embedding.cuda()
            else:
              embedding_copy = embedding.clone().detach()
             
            all_data[label.item()].append(embedding_copy)
           
        for idx, embeds in enumerate(all_data):
            anchors[idx] = (torch.stack(embeds).sum(0) / len(embeds))
                
        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []
 
    draw_umap(all_data,
              min_dist=0.8, n_components=2, n_neighbors=15,
              title="Embedding_Clusters_Train_Data")

    total_loss /= (batch_count)
    return total_loss, metrics, anchors


def test_epoch(val_loader, model, loss_fn, cuda, metrics, testName="testVideo"):
   
    global folder
    folder = testName 
    all_data = [[] for _ in range(shared_params["classes"])]
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        batch_count = len(val_loader) or 1
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target


            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

            val_loss += loss.item()

 
            for embedding, label in zip(loss_inputs[0], loss_inputs[1]):
                if cuda: 
                  embedding_copy = embedding.cpu().clone().detach()
                  embedding.cuda()
                else:
                  embedding_copy = embedding.clone().detach()
                all_data[label.item()].append(embedding_copy)
                

            for metric in metrics:
                metric(outputs, target, loss_outputs)


        draw_umap(all_data, #torch.stack(pos+ fos),
                  min_dist=0.8, n_components=2, n_neighbors=15,
                  title="Embedding_Clusters_Test_Data")
        val_loss /= batch_count
    return val_loss, metrics
