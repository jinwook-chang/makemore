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
itos_x = {i:s for s, i in stoi.items()}

stoi_y = {s:i for i, s in enumerate(chars)}
itos_y = {i:s for s, i in stoi_y.items()}
# %%
import torch

N = torch.zeros((729, 27), dtype = torch.float32)

for w in words:
    chs = ["."] + list(w) + ["."] # add special token to each word
    for i in range(0, len(chs) - 2):
        x = ''.join(chs[i:i+2])
        y = chs[i+2]
        ix1 = stoi_x[x]
        ix2 = stoi_y[y]
        N[ix1, ix2] += 1

# Now we have counter table for each characters 

P = (N+0.0001) / (N.sum(1, keepdim=True) + 0.0001) # Normalize it as probability +1 is for smoothing

out = []
is_finished = True
ch = ".."

while is_finished:
    idx = stoi_x[ch]
    row = P[idx, :]
    idx = torch.multinomial(row, 1, replacement=True).item()
    ch1 = list(ch)[1]
    ch2 = itos_y[idx]
    ch = ch1 + ch2
    if ch2 == ".":
        is_finished = False
        break
    out.append(ch2)

print(''.join(out))

# %%
# Let's define Loss fucntion

## We woud like to have likelihood which is calcualted by prob A * prob B * prob C ...
## but this likelihood is too tiny number we will transform it os log
## and log(prob A * prob B * prob C) is equal to log(prob A) + log(prob B) + log(prob C)

log_likelihood = 0
n = 0
for w in words:
    chs = ["."] + list(w) + ["."] # add special token to each word
    for i in range(0, len(chs) - 2):
        x = ''.join(chs[i:i+2])
        y = chs[i+2]
        ix1 = stoi_x[x]
        ix2 = stoi_y[y]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(f"{x}:{y} : {prob:.4f} : {logprob:.4f}")
print(f"{log_likelihood=}")
print(f"{-log_likelihood/n}")