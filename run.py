import torch
from model import GeneralizedConvSeq2Seq

device = 'cuda'

def make_model1():
    model = GeneralizedConvSeq2Seq()
    # move model and weights to GPU
    model.to(device)
    ckpt = torch.load('weights.pth', map_location=device)

    # then load model and make prediction
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model

def make_model2():
    model = GeneralizedConvSeq2Seq()
    # model and weights are on CPU
    ckpt = torch.load('weights.pth', map_location='cpu')
    model.load_state_dict(ckpt['model'])

    # move model to GPU and make predictions
    model.to(device)
    model.eval()
    return model

inputs = torch.rand((1, 3, 32, 64))
model1 = make_model1()
model2 = make_model2()

with torch.no_grad():
    # check same output backbone
    out1 = model1.backbone(inputs.to(device))
    out2 = model2.backbone(inputs.to(device))
    print(out1.shape)
    print(out2.shape)
    assert torch.equal(out1, out2)

    # flatten out1 B,C,H,W -> B,S,E
    B, E, H, W = out1.shape
    out1 = out1.reshape(B, E, H * W)                    # B, E, H * W
    out1 = out1.transpose(-2, -1)                       # B, S = H * W, E

    # flatten out2 B,C,H,W -> B,S,E
    B, E, H, W = out2.shape
    out2 = out2.reshape(B, E, H * W)                    # B, E, H * W
    out2 = out2.transpose(-2, -1)                       # B, S = H * W, E
    assert torch.equal(out1, out2)

    out1 = model1._forward_eval(out1, image_padding_mask=None)
    out2 = model2._forward_eval(out2, image_padding_mask=None)
    print(out1.shape)
    print(out2.shape)
    assert torch.equal(out1, out2), f'{out1.shape} != {out2.shape}'  # False
