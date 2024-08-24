# def read_text(): 
#     with open('../pg35.txt','r') as f:
#         lines = f.readlines()
#     print('line: {}'.format(len(lines)))
    
    
# read_text()
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a = torch.tensor([3,4],device=device)
print(a,a.device)