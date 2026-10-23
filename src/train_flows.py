import torch

def train_flow(model, train_data, test_data, epochs=500, batch_size=256, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_tens = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)
    test_tens = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)

    train_loader = torch.utils.data.DataLoader(
        train_tens, batch_size=batch_size, shuffle=True
    )

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            log_prob = model.log_prob(batch)
            log_prob = torch.clamp(log_prob, min=-20.0, max=20.0)
            loss = -log_prob.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.detach().item()

        train_losses.append(epoch_loss / len(train_loader))

        with torch.no_grad():
            model.eval()
            test_log_prob = model.log_prob(test_tens)
            test_log_prob = torch.clamp(test_log_prob, min=-20.0, max=20.0)
            test_loss = -test_log_prob.mean().item()
            test_losses.append(test_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Train = {train_losses[-1]}, Test = {test_losses[-1]}")

    return train_losses, test_losses