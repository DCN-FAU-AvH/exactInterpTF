import torch

def generateSyntheticData(N, d, n, param_norm = True):
    """
    Generate random input sequences and output sequences by the action of a transformer block.
    Inputs:
      N (int): number of sequences
      d (int): token dimension
      n (int): number of tokens per sequence
    Returns:
      inputs: tensor of shape (N, n, d)
      outputs: tensor of shape (N, 1, d) 
    """
    # Input sequences
    inputs = torch.rand(N, n, d)

    # SA layer collapsing every sequence to its center of mass with Q = K = 0, V = I, rho = 0
    Q = torch.zeros(d, d)
    K = torch.zeros(d, d)
    V = torch.eye(d, d)
    rho = torch.zeros(1)

    Q = inputs @ Q
    K = inputs @ K
    V = inputs @ V

    scores = torch.bmm(Q, K.transpose(1, 2))
    att = torch.softmax(scores, dim=-1)

    SA = rho * inputs + torch.bmm(att, V)

    # FF layer moving the centers of mass
    W = torch.rand(4, d)
    U = torch.rand(d, 4)
    b = torch.rand(4)
    eta = torch.ones(1)

    FF = eta * SA + torch.relu(SA @ U + b) @ W

    outputs = FF[:, -1:, :]  # Last point in each sequence

    if param_norm:
        exact_L2norm = torch.norm(V, p=2)**2 + torch.norm(eta, p = 2)**2 + torch.norm(W, p=2)**2 + torch.norm(U, p=2)**2 + torch.norm(b, p=2)**2
        print(f'exact l2 parameter norm squared: {exact_L2norm}')

    return inputs, outputs, exact_L2norm if param_norm else None
