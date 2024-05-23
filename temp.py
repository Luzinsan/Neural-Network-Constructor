import torchvision.datasets as ds # type: ignore
from torchvision.transforms import v2 # type: ignore
import re

df = ds.FashionMNIST('datasets')

print(df.data.shape)

transforms = v2.Compose([v2.Resize((224,244)), v2.ToImage()])
df = ds.FashionMNIST('datasets', transform=transforms)

print(df.data.shape)
 
print(transforms.transforms)

# a = [1,2,v2.Resize((224,244))]
# print(a)
# print(a.index(v2.Resize((224,244))))
rept_transforms = transforms.extra_repr() 
print(rept_transforms)
print('Resize' in rept_transforms)
print(re.search('(?<=Resizee\(size=)(\[.*\])', rept_transforms))
print(eval(re.search('(?<=Resize\(size=)(\[.*\])', rept_transforms)[0]))