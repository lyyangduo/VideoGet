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

class MaxMarginRankingLoss(nn.Module):
    def __init__(self, margin=1):
        super(MaxMarginRankingLoss, self).__init__()
        self.loss = th.nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self,x):
        n = x.size()[0] #128
        
        x1 = th.diag(x) #[128]
        x1 = x1.unsqueeze(1) #[128,1]
        x1 = x1.expand(n, n)  #Each row same value, every column the diagnal elements
        x1 = x1.contiguous().view(-1,1) #like the reshape in numpy ==>[16384,1]
        x1 = th.cat((x1,x1),0)   #[32768, 1]

        x2 = x.view(-1,1) #original matrix (16384, 1)  row first arrange
        x3 = x.transpose(0,1).contiguous().view(-1,1) #(16384, 1) column first arrange
       
        x2 = th.cat((x2,x3),0) #(32768,1)
         
        #
        max_margin = F.relu(self.margin - (x1 - x2))
        #max_margin = F.relu(self.margin + (x1 - x2))  #Yang change!

        return max_margin.mean()
    
class MaxMarginRankingLoss_Distance(nn.Module):
    def __init__(self, margin=1):
        super(MaxMarginRankingLoss_Distance, self).__init__()
        self.loss = th.nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self,x):
        n = x.size()[0] #128
        
        x1 = th.diag(x) #[128]
        x1 = x1.unsqueeze(1) #[128,1]
        x1 = x1.expand(n, n)  #Each row same value, every column the diagnal elements
        x1 = x1.contiguous().view(-1,1) #like the reshape in numpy ==>[16384,1]
        x1 = th.cat((x1,x1),0)   #[32768, 1]

        x2 = x.view(-1,1) #original matrix (16384, 1)  row first arrange
        x3 = x.transpose(0,1).contiguous().view(-1,1) #(16384, 1) column first arrange
       
        x2 = th.cat((x2,x3),0) #(32768,1)
         
        #max_margin = F.relu(self.margin - (x1 - x2))
        max_margin = F.relu(self.margin + (x1 - x2))  #Yang change!

        return max_margin.mean()
