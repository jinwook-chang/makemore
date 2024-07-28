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
chars = sorted(list(set("".join(words))))
stoi = {s:i for i, s in enumerate(chars)}
stoi["<.>"] = 26 # Special token for SOS/EOS
itos = {i:s for s, i in stoi.items()}
# %%
import torch

N = torch.zeros((27, 27), dtype = torch.float32)

for w in words:
    chs = ["<.>"] + list(w) + ["<.>"] # add special token to each word
    for ch1, ch2 in zip(chs, chs[1:]): 
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# Now we have counter table for each characters 

P = (N+1) / N.sum(1, keepdim = True) # Normalize it as probability +1 is for smoothing

out = []
is_finished = True
ch = "<.>"

while is_finished:
    idx = stoi[ch]
    row = P[idx, :]
    idx = torch.multinomial(row, 1, replacement=True).item()
    ch = itos[idx]
    if ch == "<.>":
        is_finished = False
    out.append(ch)

print(out)

# %%
# Let's define Loss fucntion

## We woud like to have likelihood which is calcualted by prob A * prob B * prob C ...
## but this likelihood is too tiny number we will transform it os log
## and log(prob A * prob B * prob C) is equal to log(prob A) + log(prob B) + log(prob C)

log_likelihood = 0
n = 0
for w in words:
    chs = ["<.>"] + list(w) + ["<.>"]
    for ch1, ch2 in zip(chs, chs[1:]): 
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(f"{ch1}{ch2} : {prob:.4f} : {logprob:.4f}")
print(f"{log_likelihood=}")
print(f"{-log_likelihood/n}")

# %%
# Let make it neural 👌👌👌

