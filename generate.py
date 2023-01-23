def sample(model):
    model.train(False)
    data = torch.zeros(sample_batch_size, obs[0]*args.block_dim * args.block_dim, obs[1]//args.block_dim, obs[2]//args.block_dim)
    data = data.cuda()
    for i in range(obs[1]//args.block_dim):
        for j in range(obs[2]//args.block_dim):
            data_v = Variable(data, volatile=True)
            alpha = torch.rand_like(data_v) #
            out_sample   = model(data_v, alpha, sample=True) #
            # out_sample = sample_op(out)
            data[:, :, i, j] = out_sample.data[:, :, i, j]
    data = data.cpu().detach().numpy()
    pixels2 = np.zeros(shape=(sample_batch_size, obs[0], obs[1], obs[2]))
    print(obs[0])
    for i in range(obs[1]//args.block_dim):
      for j in range(obs[2]//args.block_dim):
          digit = data[:, :, i, j].reshape([sample_batch_size, obs[0], args.block_dim, args.block_dim])
          pixels2[:, :, i * args.block_dim: (i + 1) * args.block_dim,
               j * args.block_dim: (j + 1) * args.block_dim] = digit

    return torch.from_numpy(pixels2).cuda()