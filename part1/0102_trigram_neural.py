# %%
# Read Data
words = open("names.txt", "r").read().splitlines()
words[:10]
# %%
# Glimpse data
len(words) # 32,033 
min([len(w) for w in words]) # 2
max([len(w) for w in words]) # 15

# %%
# make dictionary for string to idx & idx to string


from itertools import product

chars = sorted(list(set("".join(words))))

chars = chars + ["."]
stoi_x = {(c[0]+c[1]) : i for i, c in enumerate(product(chars, chars))}
itos_x = {i:s for s, i in stoi_x.items()}

stoi_y = {s:i for i, s in enumerate(chars)}
itos_y = {i:s for s, i in stoi_y.items()}
# %%
import torch
device = torch.device('cuda:0')
# %%
# Let make it neural 👌👌👌

## Create training set

xs, ys = [], []

for w in words:
    chs = ["."] + list(w) + ["."] # add special token to each word
    for i in range(0, len(chs) - 2):
        x = ''.join(chs[i:i+2])
        y = chs[i+2]
        ix1 = stoi_x[x]
        ix2 = stoi_y[y]
        xs.append(ix1)
        ys.append(ix2)



xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()


# %%
import torch.nn.functional as F

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((729, 27), generator = g, requires_grad = True).to(device)

# %%
for i in range(0,100):
    # forward pass
    xenc = F.one_hot(xs, num_classes = 729).float()
    logits = xenc @ W # log-counts
    counts = logits.exp() # counts which is equivalent to N
    probs = counts / counts.sum(1, keepdim = True)
    loss = -probs[torch.arange(num), ys].log().mean() # + 0.01*(W**2).mean() # L2 Regul

    # backward pass
    W.grad = None # set to gradient as zero
    loss.backward()
    print(loss)
    # update
    W.data += W.grad * -50
