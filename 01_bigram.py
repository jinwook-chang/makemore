# %%
words = open('names.txt', 'r').read().splitlines()
words[:10]
# %%
len(words) # 32,033 
min([len(w) for w in words]) # 2
max([len(w) for w in words]) # 15
# %%
b = {}

for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]): 
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

sorted(b.items(), key=lambda x:x[1], reverse=True)
# %%
