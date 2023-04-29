import torch


def get_device(use_gpu):
    if not use_gpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda:0")


def train_model_step(model, train_loader, val_loader, optimizer, criterion, device):
    with torch.enable_grad():
        model.train()

        # Process train set.
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
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

        # Process val set.
        val_loss = 0
        val_acc = 0
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                y = model(batch_x)
                val_loss += criterion(y, batch_y).item()
                val_acc += torch.argmax(y, axis=-1).eq(batch_y).sum().item()
            val_loss /= len(val_loader)
            val_acc /= len(val_loader.dataset)

        print(f"train_loss = {train_loss:.04f}, val_loss = {val_loss:.04f}, val_acc = {val_acc:.04f}")

        return train_loss, val_loss


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, patience, use_gpu):
    device = get_device(use_gpu)
    model = model.to(device)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print(f"epoch = {epoch + 1}/{epochs}")
        _, val_loss = train_model_step(model, train_loader, val_loader, optimizer, criterion, device)
        
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