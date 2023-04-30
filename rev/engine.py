import torch


DEVICE = None


def set_device(use_gpu):
    global DEVICE
    DEVICE = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")


def evaluate_model(model, loader, criterion):
    global DEVICE

    with torch.no_grad():
        model.eval()

        # Set initial loss and accuracy.
        loss = 0
        acc = 0

        # For each batch.
        for batch_x, batch_y in loader:

            # Compute predictions.
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            y = model(batch_x)

            # Update loss and accuracy.
            loss += criterion(y, batch_y).item()
            acc += torch.argmax(y, axis=-1).eq(batch_y).sum().item()

        # Normalize loss and accuracy.
        loss /= len(loader)
        acc /= len(loader.dataset)

        return loss, acc



def train_model_step(model, train_loader, val_loader, optimizer, criterion):
    global DEVICE

    with torch.enable_grad():
        model.train()
        optimizer.zero_grad()

        # Process train set.
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            # Process batch.
            y = model(batch_x)
            loss = criterion(y, batch_y)

            # Step optimizer.
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            # Compute loss
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Process validation set.
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        print(f"train_loss = {train_loss:.04f}, val_loss = {val_loss:.04f}, val_acc = {val_acc:.04f}")

        return train_loss, val_loss


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, patience, use_gpu):
    global DEVICE

    set_device(use_gpu)
    model = model.to(DEVICE)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print(f"epoch = {epoch + 1}/{epochs}")
        _, val_loss = train_model_step(model, train_loader, val_loader, optimizer, criterion)
        
        # ChatGPT
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"best_val_loss = {best_val_loss:.04f}, epochs_without_improvement = {epochs_without_improvement}")
        if epochs_without_improvement >= patience:
            break
    
    return model