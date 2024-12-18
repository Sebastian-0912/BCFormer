import torch

# print(torch.cuda.is_available())
# 
# print(torch.eq(torch.tensor([1, 2]), torch.tensor([1, 2])).sum().item())

# print(torch.equal(torch.tensor([1, 2]), torch.tensor([1, 2])))
# exact_match_count = 0 
# # 計算完全匹配（Exact Match）準確率
# exact_match = torch.all(torch.eq(torch.tensor([[1, 1]]), torch.tensor([[1, 2]])), dim=1)
# exact_match_count += exact_match.sum().item()  # 完全匹配樣本數
# exact_match_accuracy = exact_match_count 
# print(exact_match_accuracy)

a = torch.tensor([[1,2,3],[2,34,5]])
b = torch.tensor([[1,2,3],[2,34,5]])
d = []
d.append(a)
d.append(b)
c = torch.cat(d,dim=0)
print(c)
print(d)