import logging

os.makedirs('log', exist_ok=True)
logging.basicConfig(
    filename='log/2logs.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

logging.info('Обучение началось')


for epoch in range(EPOCH):

    print(f'Epoch: [ {epoch+1:^2} / {EPOCH} ]')
    logging.info(f'Epoch: [ {epoch+1:^2} / {EPOCH} ] started')

    # train
    train_loss = 0.0
    for xb,yb in tqdm(train_dataset, desc='Train'):

        B, N, C, H, W = xb.shape

        x = xb.view(B*N, C, H, W).to(device, non_blocking=True)
        y = yb.unsqueeze(1).repeat(1, N, 1).view(B*N, 234)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= total_train

    # valid
    valid_loss = 0.0
    targets_all, probs_all = [], []

    with torch.no_grad():

        for xb, yb in tqdm(valid_dataset, desc='Valid'):

            B, N, C, H, W = xb.shape

            x = xb.view(B*N, C, H, W).to(device, non_blocking=True)

            y = yb.unsqueeze(1).repeat(1, N, 1).view(B*N, 234)
            y = y.to(device, non_blocking=True)

            logits = model(x)

            loss = criterion(logits, y)
            valid_loss += loss.item()

            probs = torch.sigmoid(logits)

            targets_all.append(y.cpu())
            probs_all.append(probs.cpu())

    valid_loss /= total_valid

    targets = torch.cat(targets_all).numpy()
    probs_all = torch.cat(probs_all).numpy()

    best_thr = 0.5
    best_f1 = 0

    thrs = np.linspace(0.1, 0.9, 10)

    for thr in thrs:

        predicts = (probs_all > thr).astype(int)

        f1 = f1_score(
            targets,
            predicts,
            average='macro',
            zero_division=0
        )

        if best_f1 < f1:
            best_f1 = f1
            best_thr = thr

    predicts = (probs_all > best_thr).astype(int)

    f1_micro = f1_score(
        targets,
        predicts,
        average='micro',
        zero_division=0
    )

    f1_macro = f1_score(
        targets,
        predicts,
        average='macro',
        zero_division=0
    )

    print(f'     F1-micro: {green}{f1_micro:.4f}{reset}, ValidLoss: {yellow}{valid_loss:.4f}{reset}')
    print(f'     F1-macro: {f1_macro:.4f}, TrainLoss: {train_loss:.4f}')
    print(f'     threshold: {best_thr}, lr: {optimizer.param_groups[0]["lr"]}')

    logging.info(
        f'F1-micro: {f1_micro:.4f}, '
        f'F1-macro: {f1_macro:.4f}, '
        f'TrainLoss: {train_loss:.4f}, '
        f'ValidLoss: {valid_loss:.4f}, '
        f'Threshold: {best_thr:.2f}, '
        f'LR: {optimizer.param_groups[0]["lr"]}'
    )

    if f1_macro > best_metric:

        best_metric = f1_macro
        num_epoch = 0

        torch.save(model.state_dict(), 'best_multi.pth')
        print(f'     Модель {epoch+1} эпохи сохранена')
        logging.info(f'Модель {epoch+1} эпохи сохранена')

    else:

        num_epoch += 1

        print(f'     Метрика не выросла')
        logging.info('Метрика не выросла')

        if num_epoch >= max_epoch:

            stop_train = True

            print(f'     ПЛАТО - ОСТАНОВКА ОБУЧЕНИЯ ')
            logging.info('ПЛАТО - ОСТАНОВКА ОБУЧЕНИЯ')

    if stop_train:
        break