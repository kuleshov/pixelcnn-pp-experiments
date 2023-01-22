for batch_idx, (input,_) in enumerate(train_loader):
    input = input.cuda(async=True)
    input = Variable(input)
    
    alpha = torch.rand_like(input)
    output = model(input, alpha)
    loss = quantile_loss(input, alpha)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss += loss.data[0]
    if (batch_idx +1) % args.print_every == 0 : 
        deno = args.print_every * args.batch_size * np.prod(obs) * np.log(2.)
        writer.add_scalar('train/bpd', (train_loss / deno), writes)
        print('loss : {:.4f}, time : {:.4f}'.format(
            (train_loss / deno), 
            (time.time() - time_)))
        train_loss = 0.
        writes += 1
        time_ = time.time()