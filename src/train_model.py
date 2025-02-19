import torch
from torch import nn

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, Xin in enumerate(dataloader):
        X = Xin['pdf']
        y = Xin['nwords']
        pred = model(X)
        target = torch.flatten(torch.round( torch.clamp( y/100.0, min=0.0, max=29.0 ) ) ).long() 
        
        loss = loss_fn( pred, target )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct, correct_flex = 0, 0, 0

    with torch.no_grad():
        for Xin in dataloader:
            X = Xin['pdf']
            y = Xin['nwords']
            
            pred = model(X)
            target = torch.flatten(torch.round( torch.clamp( y/100.0, min=0.0, max=29.0 ) ) ).long() 
            
            ls = loss_fn( pred, target )
            er = torch.sqrt(y)
            test_loss += ls.item()

            correct += (pred.argmax(1) == target).type(torch.float).sum().item()
            correct_flex += (torch.abs(pred.argmax(1) - target) <= 1).type(torch.float).sum().item()
            

    test_loss /= num_batches
    correct /= size
    correct_flex /= size
    print(f"Exact Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f" Test +/-1 Error:\n Accuracy: {(100*correct_flex):>0.1f}%, Avg loss: {test_loss:>8f} \n")
