import torch

from socr.text.models.resSru import resSru

labels = {"": 0, "a": 1, "b": 2, "c": 3}

model = resSru(labels)
# model.load_state_dict(...)

loss = model.create_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

