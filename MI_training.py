# Write by Rodolphe Nonclercq
# October 2023
# ILLS-LIVIA
# contact : rnonclercq@gmail.com



import os
import torch
import pandas as pd
import torchvision
from torchvision.models import resnet50, ResNet50_Weights,resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
from knife import KNIFE
import torch
from tqdm import tqdm




def train(encoder,MI_model,classifier_affect,classifier_ID,loader_train,loader_test,device,lr=0.0001,lr_MI=0.01,lamb=0.,EPOCH=10,loss_affect=None,save_path=None):
    save_model_name = 'encoder_'+str(lamb)+'.pt'
    save_model_name_ID = 'ID_'+str(lamb)+'.pt'
    save_model_name_Affect = 'Affect_'+str(lamb)+'.pt'
    save_model_name_MI = 'MI_'+str(lamb)+'.pt'
    save_log_name = 'log_'+str(lamb)+'.csv'
    if loss_affect is None:
        loss_affect = torch.nn.BCELoss(reduction='sum')
    loss_ID = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.Adam(list(encoder.parameters())+list(classifier_affect.parameters())+list(classifier_ID.parameters()),lr=lr)
    optimizer_MI = torch.optim.Adam(list(MI_model.parameters()),lr=lr_MI)

    min_loss_val = None
    dic_log = {'loss_CE_train':[],'loss_CE_val':[],'loss_acc_train':[],'loss_acc_val':[],'loss_acc_ID_train': [],'MI':[]}
    count = 0 
    for epoch in range(EPOCH):
        loader_train.dataset.reset()
        loss_task_tot = 0
        elem_sum = 0
        true_response_affect = 0
        true_response_ID = 0
        MI_loss_tot = 0
        encoder.train()
        classifier_affect.train()
        classifier_ID.train()
        MI_model.train()
        loop_train = tqdm(loader_train,colour='BLUE')
        for i,pack in enumerate(loop_train):

            img_tensor = pack[0].to(device)
            pain_tensor = pack[1].float().to(device)
            ID_tensor = pack[2].to(device)
            ID_tensor_one_hot = torch.nn.functional.one_hot(ID_tensor,61).float()
            elem_sum += img_tensor.shape[0]

            #Encoding
            encoded_img  = encoder(img_tensor)

            # UPDATE MI
            loss_MI = MI_model.loss(encoded_img.detach(),ID_tensor_one_hot)
            optimizer_MI.zero_grad()
            loss_MI.backward()
            optimizer_MI.step()
            MI_loss_tot += float(loss_MI)
            
            # TASK Affect
            output = classifier_affect(encoded_img)
            loss_task_affect = loss_affect(output,pain_tensor)
            loss_task_tot += float(loss_task_affect) 
            true_response_affect += float(torch.sum(output.round() == pain_tensor))

            # Task ID
            output = classifier_ID(encoded_img.detach())
            loss_task_ID = loss_ID(output,ID_tensor) 
            true_response_ID += float(torch.sum(output.max(dim=-1)[1] == ID_tensor))

            loss =  loss_task_ID + loss_task_affect + lamb*MI_model.I(encoded_img,ID_tensor_one_hot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            

            loop_train.set_description(f"Epoch [{epoch}/{EPOCH}] training")
            loop_train.set_postfix(loss_task = loss_task_tot/elem_sum,accuracy_pain=true_response_affect/elem_sum*100,accuracy_ID=true_response_ID/elem_sum*100,MI=MI_loss_tot/elem_sum)
        
        encoder.eval()
        MI_model.eval()
        classifier_affect.eval()
        classifier_ID.eval()

        loss_task_val = 0
        elem_sum_val = 0
        true_response_affect_val  =0
        true_response_ID_val = 0
        loop_test = tqdm(loader_test,colour='GREEN')
        for pack in loop_test:
            img_tensor = pack[0].to(device)
            pain_tensor = pack[1].float().to(device)
            ID_tensor = pack[2].to(device)

            elem_sum_val += img_tensor.shape[0]

            with torch.no_grad():
                #Encoding
                encoded_img  = encoder(img_tensor)

                # TASK Affect
                output = classifier_affect(encoded_img)
                loss_task_affect_val = loss_affect(output,pain_tensor) 
                true_response_affect_val += float(torch.sum(output.round() == pain_tensor))
                loss_task_val += float(loss_task_affect_val)

                # Task ID
                output = classifier_ID(encoded_img.detach())
                loss_task_ID_val = loss_ID(output,ID_tensor) 
                true_response_ID_val += float(torch.sum(output.max(dim=-1)[1] == ID_tensor))

            
            
            loop_test.set_description(f"Test lambda {lamb}")
            loop_test.set_postfix(loss_task = loss_task_val/elem_sum_val,accuracy_pain=true_response_affect_val/elem_sum_val*100,accuracy_ID=true_response_ID_val/elem_sum_val*100)

        dic_log['loss_CE_train'].append(loss_task_tot/elem_sum)
        dic_log['MI'].append(MI_loss_tot/elem_sum)
        dic_log['loss_acc_train'].append(true_response_affect/elem_sum*100)
        dic_log['loss_CE_val'].append(loss_task_val/elem_sum_val)
        dic_log['loss_acc_val'].append(true_response_affect_val/elem_sum_val*100)
        dic_log['loss_acc_ID_train'].append(true_response_ID/elem_sum*100)
        if not save_path is None:
            #if min_loss_val is None or min_loss_val > loss_task_val/elem_sum_val:
                #torch.save(encoder.state_dict(),save_path+save_model_name)
                #torch.save(classifier_ID.state_dict(),save_path+save_model_name_ID)
                #torch.save(classifier_affect.state_dict(),save_path+save_model_name_Affect)
                #torch.save(MI_model.state_dict(),save_path+save_model_name_MI)
                #min_loss_val = loss_task_val/elem_sum_val

            dataframe = pd.DataFrame(dic_log)
            dataframe.to_csv(save_path+save_log_name)
    

class Classif(torch.nn.Module):
    def __init__(self,nb_class,softmax=True):
        super(Classif,self).__init__()

        self.fc1 = torch.nn.Linear(1000,100)
        self.fc2 = torch.nn.Linear(100,nb_class)
        self.softmax = softmax


    def forward(self,x):
        x = self.fc1(x).relu()
        if self.softmax:
            x = self.fc2(x).softmax(dim=-1)
        else:
            x = self.fc2(x).sigmoid().reshape(-1)
        return x


#%%
if __name__ == '__main__':
    save_path = None

    for lamb in [0.05 + i*0.1 for i in range(21)]:
        model_resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        model_resnet.load_state_dict(torch.load(save_path+'encoder.pt'))

        model_affect = Classif(1,False).to(device)
        model_affect.load_state_dict(torch.load(save_path+'Affect.pt'))

        model_ID = Classif(nb_ID).to(device)
        model_ID.load_state_dict(torch.load(save_path+'ID.pt'))

        arg_MI= {'zd_dim':1000, 'zc_dim':61,'hidden_state':100, 'layers':3, 'nb_mixture':10,'tri':False}
        MI = KNIFE(**arg_MI).to(device)
        MI.load_state_dict(torch.load(save_path+'MI.pt'))

        train(model_resnet,MI,model_affect,model_ID,lamb=lamb,EPOCH=20,lr=0.000005,save_path=save_path)
