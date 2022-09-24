import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASS = 3
BATCH_SIZE = 8

outputs = torch.rand((BATCH_SIZE, NUM_CLASS))          #
labels = torch.randint(0, NUM_CLASS, (BATCH_SIZE,))

# 1 CE
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)
print(loss)

# 2 CE for soft label
log_softmax = F.log_softmax(outputs, dim=1)

soft_labels_A = F.one_hot(labels, NUM_CLASS)
soft_labels_B = F.softmax(outputs, dim=1)
soft_labels = (soft_labels_A + soft_labels_B) / 2.0

# 2.1 same as convention
loss1 = - torch.sum(soft_labels_A * log_softmax) / BATCH_SIZE

# 2.2 variant
loss2 = - torch.sum(soft_labels * log_softmax) / BATCH_SIZE

print(loss1, loss2)