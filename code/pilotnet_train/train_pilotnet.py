import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

from dataset_pilotnet import PilotNetNPYDataset
from model_pilotnet import PilotNet
from config_train import CSV_PATH, NPY_ROOT, BATCH_SIZE, EPOCHS, LR

def main():
    df = pd.read_csv(CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.1, shuffle=True)

    train_df.to_csv("train_split.csv", index=False)
    val_df.to_csv("val_split.csv", index=False)

    train_ds = PilotNetNPYDataset("train_split.csv", NPY_ROOT)
    val_ds   = PilotNetNPYDataset("val_split.csv", NPY_ROOT)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PilotNet().to(device)

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for imgs, yaws in train_loader:
            imgs, yaws = imgs.to(device), yaws.to(device).unsqueeze(1)
            optim.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, yaws)
            loss.backward()
            optim.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, yaws in val_loader:
                imgs, yaws = imgs.to(device), yaws.to(device).unsqueeze(1)
                val_loss += loss_fn(model(imgs), yaws).item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Train {train_loss:.3f} | Val {val_loss:.3f}")

    torch.save(model.state_dict(), "/home/tv/TiNy_PilotNet/model/pilotnet.pth")
    print("Saved: pilotnet.pth")

if __name__ == "__main__":
    main()
