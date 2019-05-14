

def main():
    from models.enet import ENet
    from experiments.RandomShapes.RndShapesDataset import RndShapesDataset

    classesNum = len(RndShapesDataset.color_encoding)
    model = ENet(classesNum)
    import torch
    state = torch.load('./checkpoints/ENet')
    model.load_state_dict(state['state_dict'])
    # load image, resize and convert to mini-batch tensor

main()
