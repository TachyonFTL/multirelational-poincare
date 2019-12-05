import torch

def artanh(x):
    return 0.5*torch.log((1+x)/(1-x))

def p_exp_map(v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
    return torch.tanh(normv)*v/normv

def p_log_map(v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), 1e-10, 1-1e-5)
    return artanh(normv)*v/normv

def full_p_exp_map(x, v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
    sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5)
    y = torch.tanh(normv/(1-sqxnorm)) * v/normv
    return p_sum(x, y)

def p_sum(x, y):
    sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5)
    sqynorm = torch.clamp(torch.sum(y * y, dim=-1, keepdim=True), 0, 1-1e-5)
    dotxy = torch.sum(x*y, dim=-1, keepdim=True)
    numerator = (1+2*dotxy+sqynorm)*x + (1-sqxnorm)*y
    denominator = 1 + 2*dotxy + sqxnorm*sqynorm
    return numerator/denominator


def load_embeddings(embeddings_f, synset2index, debug=False):
    # print(f'loading pretrained embeddings from: {embeddings_f}')
    synset2embedding = {}

    with open(embeddings_f) as in_f:
        lines = tqdm(in_f.readlines(), desc=f'loading embeddings from {embeddings_f}', ascii=True)
        for line in lines:
            l = line.strip().split()
            if len(l) > 1:
                synset2embedding[l[0]] = np.asarray([float(item) for item in l[1:]])
        lines.set_description('loading embeddings ---done!')
    embedding_dim = len(synset2embedding[next(iter(synset2embedding))])
    embeddings = []
    num_unseen = 0
    unseen_synsets = []

    if debug:
        f = open('embedding_log', 'w')
        print([i for i in zip(synset2index, range(len(synset2index)))], file=f)
        syns = []

    for synset in synset2index:
        if debug:
            syns.append(synset)

        if synset in synset2embedding:
            embeddings.append(synset2embedding[synset])
        else:
            num_unseen += 1
            unseen_synsets.append(synset)
            if len(embeddings) == 0:
                embeddings.append(np.zeros(embedding_dim))
            else:
                embeddings.append(np.random.uniform(-0.8, 0.8, embedding_dim))

    if debug:
        print(syns, file=f)
        print(len(syns), file=f)

    embeddings.append(np.random.uniform(-0.8, 0.8, embedding_dim))

    if debug:
        print(embeddings[:300], file=f)

    embeddings = torch.from_numpy(np.asarray(embeddings))

    if debug:
        print(embeddings[:300], file=f)
        f.close()

    # print(embeddings.size())
    print(f'{num_unseen} synsets not found in embeddings')
    print(f'{len(embeddings) - num_unseen} synsets found in embeddings')
    print(unseen_synsets)
    return embeddings
