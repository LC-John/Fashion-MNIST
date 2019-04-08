To use Pytorch model, you should download the Pytorch_Net.py and Res18_accuracy_0.9264.pkl to the same folder.
Using
```python
model = torch.load('Res18_accuracy_0.9264.pkl')
```
to load the model and then you can feed B×C×W×H shape tensor to the model, it will ouput a B×10 tensor.
