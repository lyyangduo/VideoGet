# Copyright 2018 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from loupe import NetVLAD
import numpy as np
from torch.autograd import Function

class Net_Fuse(nn.Module):
    def __init__(self, video_modality_dim, text_dim_dic, audio_cluster=8,  text_cluster=32):
        super(Net_Fuse, self).__init__()
        
        self.audio_pooling = NetVLAD(feature_size=video_modality_dim['audio'][1],
                cluster_size=audio_cluster)
        self.text_pooling = NetVLAD(feature_size=text_dim_dic['text'][0],
                cluster_size=text_cluster)
        print (self.audio_pooling.out_dim)
        print (self.text_pooling.out_dim)
        
        #print (self.text_pooling)
        #print (self.text_pooling.out_dim)

        self.mee_fuse = MEE_FUSE(video_modality_dim, self.text_pooling.out_dim,text_dim_dic)

    def forward(self, text, video, ind, conf=True):

        aggregated_video = {}
        
        aggregated_video['audio'] = self.audio_pooling(video['audio'])
        aggregated_video['face'] = video['face'] 
        aggregated_video['motion'] = video['motion']
        aggregated_video['visual'] = video['visual']
        
        text = self.text_pooling(text)

        return self.mee_fuse(text, aggregated_video, ind, conf)
    
    
class Net_Fuse_Same(nn.Module):
    def __init__(self, video_modality_dim, text_dim_dic, audio_cluster=8,  text_cluster=32):
        super(Net_Fuse_Same, self).__init__()
        
        self.audio_pooling = NetVLAD(feature_size=video_modality_dim['audio'][1],
                cluster_size=audio_cluster)
        self.text_pooling = NetVLAD(feature_size=text_dim_dic['text'][0],
                cluster_size=text_cluster)
        print (self.audio_pooling.out_dim)
        print (self.text_pooling.out_dim)
        
        #print (self.text_pooling)
        #print (self.text_pooling.out_dim)

        self.mee_fuse_same = MEE_FUSE_Same(video_modality_dim, self.text_pooling.out_dim,text_dim_dic)

    def forward(self, text, video, ind, conf=True):

        aggregated_video = {}
        
        aggregated_video['audio'] = self.audio_pooling(video['audio'])
        aggregated_video['face'] = video['face'] 
        aggregated_video['motion'] = video['motion']
        aggregated_video['visual'] = video['visual']
        
        text = self.text_pooling(text)

        return self.mee_fuse_same(text, aggregated_video, ind, conf)

    
class Net_Fuse_Cat(nn.Module):
    def __init__(self, video_modality_dim, text_dim_dic, audio_cluster=8,  text_cluster=32):
        super(Net_Fuse_Cat, self).__init__()
        
        self.audio_pooling = NetVLAD(feature_size=video_modality_dim['audio'][1],
                cluster_size=audio_cluster)
        self.text_pooling = NetVLAD(feature_size=text_dim_dic['text'][0],
                cluster_size=text_cluster)
        print (self.audio_pooling.out_dim)
        print (self.text_pooling.out_dim)
        
        #print (self.text_pooling)
        #print (self.text_pooling.out_dim)

        self.mee_fuse_cat = MEE_FUSE_Cat(video_modality_dim, self.text_pooling.out_dim,text_dim_dic)

    def forward(self, text, video, ind, conf=True):

        aggregated_video = {}
        
        aggregated_video['audio'] = self.audio_pooling(video['audio'])
        aggregated_video['face'] = video['face'] 
        aggregated_video['motion'] = video['motion']
        aggregated_video['visual'] = video['visual']
        
        text = self.text_pooling(text)

        return self.mee_fuse_cat(text, aggregated_video, ind, conf)

    
    

class MEE_FUSE(nn.Module):
    def __init__(self, video_modality_dim, text_dim,text_dim_dic):
        super(MEE_FUSE, self).__init__()

        m = list(video_modality_dim.keys()) #Yang add

        self.m = m
        
        self.video_GU = nn.ModuleList([Gated_Embedding_Unit_SameOut(video_modality_dim[m[i]][0],
            video_modality_dim[m[i]][1],video_modality_dim[m[i]][2]) for i in range(len(m))])

        #self.text_GU = nn.ModuleList([Gated_Embedding_Unit_SameOut(text_dim,
        #    video_modality_dim[m[i]][1],video_modality_dim[m[i]][2]) for i in range(len(m))])
        
        self.text_GU = Gated_Embedding_Unit_SameOut(text_dim,
            text_dim_dic['text'][1],text_dim_dic['text'][2])

        self.moe_fc = nn.Linear(text_dim, len(video_modality_dim))
    

    def forward(self, text, video, ind, conf=True):

        text_embd = {}

        for i, l in enumerate(self.video_GU):
            video[self.m[i]] = l(video[self.m[i]]) #videodict_keys(['audio', 'face', 'motion', 'visual'])

#         for i, l in enumerate(self.text_GU):
#             text_embd[self.m[i]] = l(text) #videodict_keys(['audio', 'face', 'motion', 'visual'])
        text_embd =self.text_GU(text)
        #print (text_embd.shape)

        
        #MOE weights computation + normalization ------------
        moe_weights = self.moe_fc(text)  #(128,4)
        moe_weights = F.softmax(moe_weights, dim=1)

        available_m = np.zeros(moe_weights.size())  #(128,4)

        i = 0
        for m in video:
            #print (m)
            available_m[:,i] = ind[m] #Each iteration for one column ind ==sequence  
                                      # [audio/face/motion/visual]
            i += 1
            
        #print (available_m.shape)
        #print (i)

        available_m = th.from_numpy(available_m).float()
        available_m = Variable(available_m.cuda())

        moe_weights = available_m*moe_weights #[128,4]
        
        # If there is no data on that dimension ==>delete that weight
        #print (moe_weights.shape)

        norm_weights = th.sum(moe_weights, dim=1) #[128]
        #print (norm_weights)
        norm_weights = norm_weights.unsqueeze(1) #[128,1]
        #print (norm_weights.shape)
        moe_weights = th.div(moe_weights, norm_weights) #[128,4]  ==Each row sum to 1
        #print (moe_weights)

        #MOE weights computation + normalization ------ DONE

        if conf:
            conf_matrix = Variable(th.zeros(len(text),len(text)).cuda()) #(128,128)
            fuse_video_feature = Variable(th.zeros(len(text),256).cuda()) 
            #print (conf_matrix.shape) 
            i = 0
            #print (text_embd.shape)
            for m in video:
                #print (m)
                #print (text_embd[m].shape)
                #video[m] = video[m].transpose(0,1)

                #i=0, video['audio']=(128,128), textembed['audio']=(128,128)
                #i=1, video['face']=(128,128), textembed['face']=(128,128)
                #video['motion']=(1024,128), textembed['audio']=(128,1024)
                #video['visual']=(2048,128), textembed['audio']=(128,2048)

                fuse_video_feature += moe_weights[:,i:i+1]*video[m]
                #print (fuse_video_feature.shape)
                
                i += 1
            
            conf_matrix=th.sum((text_embd.unsqueeze(1)-fuse_video_feature.unsqueeze(0))**2,2)
            
            return conf_matrix
        else:
            i = 0
            scores = Variable(th.zeros(len(text)).cuda())
            for m in video:
                text_embd[m] = moe_weights[:,i:i+1]*text_embd[m]*video[m]
                scores += th.sum(text_embd[m], dim=-1)
                i += 1
             
            return scores   
        
        

class MEE_FUSE_Same(nn.Module):
    def __init__(self, video_modality_dim, text_dim,text_dim_dic):
        super(MEE_FUSE_Same, self).__init__()

        m = list(video_modality_dim.keys()) #Yang add

        self.m = m
        
        self.video_GU = nn.ModuleList([Gated_Embedding_Unit_SameOut(video_modality_dim[m[i]][0],
            video_modality_dim[m[i]][1],video_modality_dim[m[i]][2]) for i in range(len(m))])

        #self.text_GU = nn.ModuleList([Gated_Embedding_Unit_SameOut(text_dim,
        #    video_modality_dim[m[i]][1],video_modality_dim[m[i]][2]) for i in range(len(m))])
        
        self.text_GU = Gated_Embedding_Unit_SameOut(text_dim,
            text_dim_dic['text'][1],text_dim_dic['text'][2])

        self.moe_fc = nn.Linear(text_dim, len(video_modality_dim))
    

    def forward(self, text, video, ind, conf=True):

        text_embd = {}

        for i, l in enumerate(self.video_GU):
            video[self.m[i]] = l(video[self.m[i]]) #videodict_keys(['audio', 'face', 'motion', 'visual'])

#         for i, l in enumerate(self.text_GU):
#             text_embd[self.m[i]] = l(text) #videodict_keys(['audio', 'face', 'motion', 'visual'])
        text_embd =self.text_GU(text)
        #print (text_embd.shape)

        
        #MOE weights computation + normalization ------------
        #moe_weights = self.moe_fc(text)  #(128,4)
        
        moe_weights_temp=np.zeros((text_embd.shape[0],4))+0.25
        moe_weights=th.from_numpy(moe_weights_temp).float()
        moe_weights = Variable(moe_weights.cuda())
        #moe_weights_tensor = th.from_numpy(moe_weights).float().to(device)
        
        moe_weights = F.softmax(moe_weights, dim=1)

        available_m = np.zeros(moe_weights.size())  #(128,4)

        i = 0
        for m in video:
            #print (m)
            available_m[:,i] = ind[m] #Each iteration for one column ind ==sequence  
                                      # [audio/face/motion/visual]
            i += 1
            
        #print (available_m.shape)
        #print (i)

        available_m = th.from_numpy(available_m).float()
        available_m = Variable(available_m.cuda())

        moe_weights = available_m*moe_weights #[128,4]
        
        # If there is no data on that dimension ==>delete that weight
        #print (moe_weights.shape)

        norm_weights = th.sum(moe_weights, dim=1) #[128]
        #print (norm_weights)
        norm_weights = norm_weights.unsqueeze(1) #[128,1]
        #print (norm_weights.shape)
        moe_weights = th.div(moe_weights, norm_weights) #[128,4]  ==Each row sum to 1
        #print (moe_weights)

        #MOE weights computation + normalization ------ DONE

        if conf:
            conf_matrix = Variable(th.zeros(len(text),len(text)).cuda()) #(128,128)
            fuse_video_feature = Variable(th.zeros(len(text),256).cuda()) 
            #print (conf_matrix.shape) 
            i = 0
            #print (text_embd.shape)
            for m in video:
                #print (m)
                #print (text_embd[m].shape)
                #video[m] = video[m].transpose(0,1)

                #i=0, video['audio']=(128,128), textembed['audio']=(128,128)
                #i=1, video['face']=(128,128), textembed['face']=(128,128)
                #video['motion']=(1024,128), textembed['audio']=(128,1024)
                #video['visual']=(2048,128), textembed['audio']=(128,2048)

                fuse_video_feature += moe_weights[:,i:i+1]*video[m]
                #print (fuse_video_feature.shape)
                
                i += 1
            
            conf_matrix=th.sum((text_embd.unsqueeze(1)-fuse_video_feature.unsqueeze(0))**2,2)
            
            return conf_matrix
        else:
            i = 0
            scores = Variable(th.zeros(len(text)).cuda())
            for m in video:
                text_embd[m] = moe_weights[:,i:i+1]*text_embd[m]*video[m]
                scores += th.sum(text_embd[m], dim=-1)
                i += 1
             
            return scores   
    
    
    
class MEE_FUSE_Cat(nn.Module):
    def __init__(self, video_modality_dim, text_dim,text_dim_dic):
        super(MEE_FUSE_Cat, self).__init__()

        m = list(video_modality_dim.keys()) #Yang add

        self.m = m
        
        self.video_GU = nn.ModuleList([Gated_Embedding_Unit_SameOut_Text(video_modality_dim[m[i]][0],
            video_modality_dim[m[i]][1],video_modality_dim[m[i]][2]) for i in range(len(m))])

        #self.text_GU = nn.ModuleList([Gated_Embedding_Unit_SameOut(text_dim,
        #    video_modality_dim[m[i]][1],video_modality_dim[m[i]][2]) for i in range(len(m))])
        
        self.text_GU = Gated_Embedding_Unit_SameOut(text_dim,
            text_dim_dic['text'][1],text_dim_dic['text'][2])

        self.moe_fc = nn.Linear(text_dim, len(video_modality_dim))
    

    def forward(self, text, video, ind, conf=True):

        text_embd = {}

#         for i, l in enumerate(self.text_GU):
#             text_embd[self.m[i]] = l(text) #videodict_keys(['audio', 'face', 'motion', 'visual'])
        text_embd =self.text_GU(text)
    
    
        for i, l in enumerate(self.video_GU):
            video[self.m[i]] = l(video[self.m[i]], text_embd) #videodict_keys(['audio', 'face', 'motion', 'visual'])


        #print (text_embd.shape)

        
        #MOE weights computation + normalization ------------
        moe_weights = self.moe_fc(text)  #(128,4)
        
        #moe_weights_temp=np.zeros((text_embd.shape[0],4))+0.25
        #moe_weights=th.from_numpy(moe_weights_temp).float()
        #moe_weights = Variable(moe_weights.cuda())
        #moe_weights_tensor = th.from_numpy(moe_weights).float().to(device)
        
        moe_weights = F.softmax(moe_weights, dim=1)

        available_m = np.zeros(moe_weights.size())  #(128,4)

        i = 0
        for m in video:
            #print (m)
            available_m[:,i] = ind[m] #Each iteration for one column ind ==sequence  
                                      # [audio/face/motion/visual]
            i += 1
            
        #print (available_m.shape)
        #print (i)

        available_m = th.from_numpy(available_m).float()
        available_m = Variable(available_m.cuda())

        moe_weights = available_m*moe_weights #[128,4]
        
        # If there is no data on that dimension ==>delete that weight
        #print (moe_weights.shape)

        norm_weights = th.sum(moe_weights, dim=1) #[128]
        #print (norm_weights)
        norm_weights = norm_weights.unsqueeze(1) #[128,1]
        #print (norm_weights.shape)
        moe_weights = th.div(moe_weights, norm_weights) #[128,4]  ==Each row sum to 1
        #print (moe_weights)

        #MOE weights computation + normalization ------ DONE

        if conf:
            conf_matrix = Variable(th.zeros(len(text),len(text)).cuda()) #(128,128)
            fuse_video_feature = Variable(th.zeros(len(text),256).cuda()) 
            #print (conf_matrix.shape) 
            i = 0
            #print (text_embd.shape)
            for m in video:
                #print (m)
                #print (text_embd[m].shape)
                #video[m] = video[m].transpose(0,1)

                #i=0, video['audio']=(128,128), textembed['audio']=(128,128)
                #i=1, video['face']=(128,128), textembed['face']=(128,128)
                #video['motion']=(1024,128), textembed['audio']=(128,1024)
                #video['visual']=(2048,128), textembed['audio']=(128,2048)

                fuse_video_feature += moe_weights[:,i:i+1]*video[m]
                #print (fuse_video_feature.shape)
                
                i += 1
            
            conf_matrix=th.sum((text_embd.unsqueeze(1)-fuse_video_feature.unsqueeze(0))**2,2)
            
            return conf_matrix
        else:
            i = 0
            scores = Variable(th.zeros(len(text)).cuda())
            for m in video:
                text_embd[m] = moe_weights[:,i:i+1]*text_embd[m]*video[m]
                scores += th.sum(text_embd[m], dim=-1)
                i += 1
             
            return scores   
    
    
    
class MEE(nn.Module):
    def __init__(self, video_modality_dim, text_dim):
        super(MEE, self).__init__()

        m = list(video_modality_dim.keys()) #Yang add

        self.m = m
        
        self.video_GU = nn.ModuleList([Gated_Embedding_Unit(video_modality_dim[m[i]][0],
            video_modality_dim[m[i]][1]) for i in range(len(m))])

        self.text_GU = nn.ModuleList([Gated_Embedding_Unit(text_dim,
            video_modality_dim[m[i]][1]) for i in range(len(m))])

        self.moe_fc = nn.Linear(text_dim, len(video_modality_dim))
    

    def forward(self, text, video, ind, conf=True):

        text_embd = {}

        for i, l in enumerate(self.video_GU):
            video[self.m[i]] = l(video[self.m[i]]) #videodict_keys(['audio', 'face', 'motion', 'visual'])

        for i, l in enumerate(self.text_GU):
            text_embd[self.m[i]] = l(text) #videodict_keys(['audio', 'face', 'motion', 'visual'])


        
        #MOE weights computation + normalization ------------
        moe_weights = self.moe_fc(text)  #(128,4)
        moe_weights = F.softmax(moe_weights, dim=1)

        available_m = np.zeros(moe_weights.size())  #(128,4)

        i = 0
        for m in video:
            print (m)
            available_m[:,i] = ind[m] #Each iteration for one column ind ==sequence  
                                      # [audio/face/motion/visual]
            i += 1
            
        #print (available_m.shape)
        #print (i)

        available_m = th.from_numpy(available_m).float()
        available_m = Variable(available_m.cuda())

        moe_weights = available_m*moe_weights #[128,4]
        
        # If there is no data on that dimension ==>delete that weight
        #print (moe_weights.shape)

        norm_weights = th.sum(moe_weights, dim=1) #[128]
        #print (norm_weights)
        norm_weights = norm_weights.unsqueeze(1) #[128,1]
        #print (norm_weights.shape)
        moe_weights = th.div(moe_weights, norm_weights) #[128,4]  ==Each row sum to 1
        #print (moe_weights)

        #MOE weights computation + normalization ------ DONE

        if conf:
            conf_matrix = Variable(th.zeros(len(text),len(text)).cuda()) #(128,128)
            #print (conf_matrix.shape) 
            i = 0
            for m in video:
                print (i)
                video[m] = video[m].transpose(0,1)
                #i=0, video['audio']=(128,128), textembed['audio']=(128,128)
                #i=1, video['face']=(128,128), textembed['face']=(128,128)
                #video['motion']=(1024,128), textembed['audio']=(128,1024)
                #video['visual']=(2048,128), textembed['audio']=(128,2048)
                print (moe_weights[:,i:i+1].shape)
                conf_matrix += moe_weights[:,i:i+1]*th.matmul(text_embd[m], video[m])
                
                i += 1

            return conf_matrix
        else:
            i = 0
            scores = Variable(th.zeros(len(text)).cuda())
            for m in video:
                text_embd[m] = moe_weights[:,i:i+1]*text_embd[m]*video[m]
                scores += th.sum(text_embd[m], dim=-1)
                i += 1
             
            return scores

class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Gated_Embedding_Unit, self).__init__()

        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)
  
    def forward(self,x):
        
        x = self.fc(x)
        x = self.cg(x)
        x = F.normalize(x)

        return x

class Gated_Embedding_Unit_SameOut(nn.Module):
    def __init__(self, input_dimension, output_dimension, output_dimension_final):
        super(Gated_Embedding_Unit_SameOut, self).__init__()

        self.fc1 = nn.Linear(input_dimension, output_dimension)
        self.fc2 = nn.Linear(output_dimension, output_dimension_final)
        self.cg = Context_Gating(output_dimension_final)
  
    def forward(self,x):
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.cg(x)
        x = F.normalize(x)

        return x    
    

class Context_Gating(nn.Module):
    def __init__(self, dimension, add_batch_norm=True):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)
        
    def forward(self,x):
        x1 = self.fc(x)

        if self.add_batch_norm:
            x1 = self.batch_norm(x1) 

        x = th.cat((x, x1), 1)
        
        return F.glu(x,1)

class Gated_Embedding_Unit_SameOut_Text(nn.Module):
    def __init__(self, input_dimension, output_dimension, output_dimension_final):
        super(Gated_Embedding_Unit_SameOut_Text, self).__init__()

        self.fc1 = nn.Linear(input_dimension, output_dimension)
        self.fc2 = nn.Linear(output_dimension, output_dimension_final)
        self.cg = Context_Gating_Text(output_dimension_final)
  
    def forward(self,x,x_text):
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.cg(x, x_text)
        x = F.normalize(x)

        return x    
    

class Context_Gating_Text(nn.Module):
    def __init__(self, dimension, add_batch_norm=True):
        super(Context_Gating_Text, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.fc_text = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)
        
    def forward(self,x,x_text):
        x1 = self.fc(x)
        x2 = self.fc_text(x_text)
        
        x3=x1+x2

        if self.add_batch_norm:
            x3 = self.batch_norm(x3) 
            
        
        x = th.cat((x, x3), 1)
        
        return F.glu(x,1)



