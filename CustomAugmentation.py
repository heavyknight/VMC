from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import numpy as np
import random
from torchvision.transforms import functional as T
import math

class Identity(object):
  def __call__(self, image, label):
    return image, label
  
  def __repr__(self):
    return '[Custom] ' + self.__class__.__name__ + '()'

class IdentityRGB(object):
  def __call__(self, image):
    return image
  
  def __repr__(self):
    return '[Custom] ' + self.__class__.__name__ + '()'

class Compose(object):
  def __init__(self, transform_list):
    self.transforms = transform_list
    
  def __call__(self, image, label):
    for t in self.transforms:
      image, label = t(image, label)
      
    return image, label
    
  def __repr__(self):
    format_string = '[Custom] ' + self.__class__.__name__ + '('
    for t in self.transforms:
      format_string += '\n'
      format_string += f'    {format(t)}'
      format_string += '\n)'
    return format_string


class ComposeChooseOne(object):
  def __init__(self, transform_list):
    self.transforms = transform_list
    self.length = len(transform_list)
    
  def get_one_index(self):
    idx = random.randint(0, self.length - 1)
    return idx
    
  def __call__(self, image, label):
    idx = self.get_one_index()
    t = self.transforms[idx]
    return t(image, label)
    
  def __repr__(self):
    format_string = '[Custom] ' + self.__class__.__name__ + '('
    for t in self.transforms:
      format_string += '\n'
      format_string += f'    {format(t)}'
      format_string += '\n)'
    return format_string


class ComposeOnlyRGB(object):
  '''only process RGB Image'''
  def __init__(self, transform_list):
    self.transforms = transform_list
    
  def __call__(self, image, label):
    for t in self.transforms:
      image = t(image)
    return image, label
    
  def __repr__(self):
    format_string = '[Custom] ' + self.__class__.__name__ + '('
    for t in self.transforms:
      format_string += '\n'
      format_string += f'    {format(t)}'
      format_string += '\n)'
    return format_string


class ComposeOnlyRGBChooseOne(object):
  '''only process RGB Image'''
  
  def __init__(self, transform_list):
    self.transforms = transform_list
    self.length = len(transform_list)
    
  def get_one_index(self):
    if self.length == 1:
      return 0
    idx = random.randint(0, self.length - 1)
    return idx
    
  def __call__(self, image, label):
    idx = self.get_one_index()
    t = self.transforms[idx]
    image = t(image)
    return image, label
    
  def __repr__(self):
    format_string = '[Custom] ' + self.__class__.__name__ + '('
    for t in self.transforms:
      format_string += '\n'
      format_string += f'    {format(t)}'
      format_string += '\n)'
    return format_string


class Resize(object):
  '''input is a tuple (size1,size2)
  process PIL images'''
  
  def __init__(self, size):
    self.size = size
  
  def __call__(self, image, label):
    image = image.resize(self.size)
    label = label.resize(self.size, resample=Image.NEAREST)
    return image, label
    
  def __repr__(self):
    format_string = '[Custom] ' + self.__class__.__name__ + f'(size={self.size})'
    return format_string


class RandomRotate(object):
  '''input angle is an integer'''
  
  def __init__(self, angle, expand=True):
    self.angleRange = angle
    self.expand = expand
    
  def get_random_angle(self):
        return random.randint(-self.angleRange, self.angleRange)
        
  def __call__(self, image, label):
    angle = self.get_random_angle()
    image = image.rotate(angle, resample=Image.BILINEAR, expand=self.expand)
    label = label.rotate(angle, resample=Image.NEAREST, expand=self.expand)
    return image, label
    
  def __repr__(self):
    format_string = '[Custom] ' + self.__class__.__name__ + f'(angle={self.angleRange}, expand={self.expand})'
    return format_string


class RandomCrop(object):
  '''Crop same region of two images
  input image are two tuples or to ints, all values positives'''
  
  def __init__(self, wide, high):
    self.wide = wide
    self.high = high
    
  def get_size(self):
    if isinstance(self.wide, tuple):
      wide = random.randint(*self.wide)
    else:
      wide = self.wide
    if isinstance(self.high, tuple):
      high = random.randint(*self.high)
    else:
      high = self.high
    return wide, high
  
  def get_coordenates(self, size):
    wide, high = self.get_size()
    s = size[0] - wide
    p = size[1] - high
    top = random.randint(min(0, s), max(0, s))
    left = random.randint(min(0, p), max(0, p))
    bottom = top + wide
    right = left + high
    return (left, top, right, bottom)
    
  def __call__(self, image, label):
    size = image.size
    region = self.get_coordenates(size)
    return image.crop(region), label.crop(region)
    
  def __repr__(self):
    format_string = '[Custom] ' + self.__class__.__name__ + f'(wide={self.wide}, high={self.high})'
    return format_string


class RandomR90(object):
  '''both images are rotate with 0, 90, 180 OR 270'''
  
  def __call__(self, image, label):
    angle = 90 * random.randint(0, 3)
    image = image.rotate(angle)
    label = label.rotate(angle, resample=Image.NEAREST)
    return image, label
    
  def __repr__(self):
    format_string = '[Custom] ' + self.__class__.__name__ + '()'
    return format_string


class RandomVerticalFlip(object):
  '''p is de prob of flip verticaly and is between 0.0 and 1.0'''
  def __init__(self, p=0.5):
    self.p = np.clip(p, 0.0, 1.0)
    
  def __call__(self, image, label):
    if self.p < random.random():
      return image, label
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return image, label
    
  def __repr__(self):
    format_string = '[Custom] ' + self.__class__.__name__ + f'(p={self.p})'
    return format_string


class RandomHorizontalFlip(object):
  '''p is de prob of flip verticaly and is between 0.0 and 1.0'''
  def __init__(self, p=0.5):
    self.p = np.clip(p, 0.0, 1.0)
    
  def __call__(self, image, label):
    if self.p < random.random():
      return image, label
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    label = label.transpose(Image.FLIP_TOP_BOTTOM)
    return image, label
    
  def __repr__(self):
    format_string = '[Custom] ' + self.__class__.__name__ + f'(p={self.p})'
    return format_string

class ToTensor(object):
  def __call__(self, image, label):
    image = T.to_tensor(image)
    label = T.to_tensor(label)
    return image, label
    
  def __repr__(self):
    format_string = '[Custom] ' + self.__class__.__name__ + '()'
    return format_string


class CartesianIlumination(object):
  '''Imput are PIL images'''
  def __init__(self, p=1.0):
    self.p = np.clip(p,0.0,1.0)
    
  def get_ilumination_map(self, size):
    w, h = size
    a, b = (0.4, 1.1) if random.random() < 0.5 else (1.1, 0.4)
    if random.random() < 0.5:
      y = np.linspace(a, b, w).reshape(1, w)
      x = np.ones((h, 1))
    else:
      y = np.ones((1, w))
      x = np.linspace(a, b, h).reshape(h, 1)
    i_map = np.matmul(x, y)[:, :, np.newaxis]
    
    return i_map
  
  def __call__(self, image):
    if self.p < random.random():
      return image
    size = image.size
    light_map = self.get_ilumination_map(size)
    image = np.uint8(np.clip(light_map * image, 0.0, 255.0))
    return Image.fromarray(image)
    
  def __repr__(self):
    format_string = '[Custom] ' + self.__class__.__name__ + f'(p={self.p})'
    return format_string


class RadialIlumination(object):
  '''Imput are PIL images'''
  def __init__(self, p=1.0):
    self.p = np.clip(p,0.0,1.0)
    
  def get_ilumination_map(self, size):
    w, h = size
    i_map = None
    l = max(w, h)
    t = np.linspace(-2.0, 2.0, l)
    t = 1.15*np.exp(-t * t / 4.0)
    y = t[(l - w) // 2:(l + w) // 2].reshape(1, w)
    x = t[(l - h) // 2:(l + h) // 2].reshape(h, 1)
    i_map = np.matmul(x, y)[:, :, np.newaxis]
    
    return i_map
    
  def __call__(self, image):
    if self.p < random.random():
      return image
    size = image.size
    light_map = self.get_ilumination_map(size)
    image = np.uint8(np.clip(light_map * image, 0.0, 255.0))
    return Image.fromarray(image)
    
  def __repr__(self):
    format_string = '[Custom] ' + self.__class__.__name__ + f'(p={self.p})'
    return format_string


class GaussianNoise(object):
  def __init__(self, p=1.0, mean=0.0, sigma=4.47):
    self.p = np.clip(p, 0.0, 1.0)
    self.mean = mean
    self.sigma = sigma
    
  def __call__(self, image):
    if self.p < random.random():
      return image
    w, h = image.size
    gauss = np.random.normal(self.mean, self.sigma, (w, h, 3))
    image = np.uint8(np.clip(image + gauss, 0.0, 255.0))
    return Image.fromarray(image)
    
  def __repr__(self):
    format_string = '[Custom] ' + self.__class__.__name__ + f'(p={self.p} , mean={self.mean}, sigma={self.sigma})'
    return format_string


class SpeckleNoise(object):
  def __init__(self, p=1.0, s=0.05):
    self.p = np.clip(p, 0.0, 1.0)
    self.s = s
    
  def __call__(self, image):
    if self.p < random.random():
      return image
    w, h = image.size
    speckle = self.s * np.random.randn(w, h, 3)
    image += image * speckle
    image = np.uint8(np.clip(image, 0.0, 255.0))
    return Image.fromarray(image)
    
  def __repr__(self):
    format_string = '[Custom] ' + self.__class__.__name__ + f'(p={self.p}, s={self.s})'
    return format_string


class SaltPepperNoise(object):
  def __init__(self, p=1.0, pp=0.5, amount=8e-3):
    self.p = np.clip(p, 0.0, 1.0)
    self.pp = np.clip(pp, 0.0, 1.0)
    self.amount = amount
    
  def __call__(self, image):
    if self.p < random.random():
      return image
    w, h = image.size
    img = np.copy(image)
    # Salt mode
    num_salt = np.ceil(self.amount * w * h * self.pp)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in (w, h)]
    img[tuple(coords)] = 255
    # Pepper mode
    num_pepper = np.ceil(self.amount * w * h * (1. - self.pp))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in (w, h)]
    img[tuple(coords)] = 0
    
    image = np.uint8(img)
    return Image.fromarray(image)
    
  def __repr__(self):
    format_string = '[Custom] ' + self.__class__.__name__ + f'(p={self.p}, pp={self.pp}, amount={self.amount})'
    return format_string


class HairC(object):
  def __init__(self, p=1.0):
    self.p = np.clip(p, 0.0, 1.0)
    self.colors = [(9, 8, 6), (44, 34, 43), (59, 48, 36), (78, 67, 63), (80, 69, 69), (106, 78, 66),
                   (85, 72, 56), (167, 133, 106), (194, 151, 128), (220, 208, 186), (222, 188, 153),
                   (151, 121, 97), (230, 206, 168), (229, 200, 168), (165, 137, 70), (145, 85, 61),
                   (83, 61, 59), (113, 99, 90), (183, 166, 158), (214, 196, 194), (250, 240, 190),
                   (202, 191, 177), (141, 74, 67), (181, 82, 57)]
                   
  def __call__(self, image):
    if self.p < random.random():
      return image
    hair_color = self.colors[random.randint(0, 23)]
    draw = ImageDraw.Draw(image)
    idx = random.randint(0, 23)
    bx = random.randint(50, 400)
    by = int(bx * np.random.uniform(0.9, 1.1))
    w, h = image.size
    a0 = random.randint(0, 360)
    l0 = random.randint(50, 100)
    
    for i in range(80):
      x1 = random.randint(-int(0.3 * w), int(1.3 * w))
      y1 = random.randint(-int(0.3 * h), int(1.3 * h))
      
      x2 = x1 + bx + random.randint(-5, 25)
      y2 = y1 + by + random.randint(-5, 25)
      
      a1 = a0 + random.randint(-20, 20)
      a2 = a1 + l0 + random.randint(-20, 20)
      
      draw.arc((x1, y1, x2, y2), start=a1, end=a2, fill=hair_color, width=1)
      
      return image
      
    def __repr__(self):
      return '[Custom] ' + self.__class__.__name__ + f'(p={self.p})'


class HairS(object):
  def __init__(self, p=1.0):
    self.p = np.clip(p, 0.0, 1.0)
    self.colors = [(9, 8, 6), (44, 34, 43), (59, 48, 36), (78, 67, 63), (80, 69, 69), (106, 78, 66),
                   (85, 72, 56), (167, 133, 106), (194, 151, 128), (220, 208, 186), (222, 188, 153),
                   (151, 121, 97), (230, 206, 168), (229, 200, 168), (165, 137, 70), (145, 85, 61),
                   (83, 61, 59), (113, 99, 90), (183, 166, 158), (214, 196, 194), (250, 240, 190),
                   (202, 191, 177), (141, 74, 67), (181, 82, 57)]
                   
  def __call__(self, image):
    if self.p < random.random():
      return image
    hair_color = self.colors[random.randint(0, 23)]
    draw = ImageDraw.Draw(image)
    idx = random.randint(0, 23)
    a0 = 2 * np.pi * np.random.uniform()
    l0 = random.randint(20, 100)
    w, h = image.size
    amount = int(2000 / l0)
    
    for i in range(amount):
      xc = random.randint(-int(0.1 * w), int(1.1 * w))
      yc = random.randint(-int(0.1 * h), int(1.1 * h))
      
      ax = a0 + np.random.uniform(-0.3, 0.3)
      ay = a0 + np.random.uniform(-0.3, 0.3)
      
      l = l0 * np.random.uniform(0.9, 1.1)
      vx, vy = np.cos(ax), np.sin(ay)
      
      lx = int(l * vx)
      ly = int(l * vy)
      
      x1 = xc - lx
      y1 = yc - ly
      
      x2 = xc + lx
      y2 = yc + ly
      
      draw.line((x1, y1, x2, y2), fill=hair_color, width=1)
      
      return image
      
    def __repr__(self):
      return '[Custom] ' + self.__class__.__name__ + f'(p={self.p})'


class FilterBlur(object):
  def __init__(self, p=1.0, alpha=(0.5, 1.0)):
    self.p = np.clip(p, 0.0, 1.0)
    self._a = alpha
    
  def get_a(self):
    if len(self._a) == 2:
      return random.uniform(*self._a)
    else:
      return self._a
      
  def __call__(self, image):
    if self.p < random.random():
      return image
    _a = self.get_a()
    return image.filter(ImageFilter.GaussianBlur(_a))
    
  def __repr__(self):
    return '[Custom] ' + self.__class__.__name__ + f'(p={self.p}, alpha={self._a})'


class FilterSharp(object):
  def __init__(self, p=1.0, alpha=(0.5, 1.0)):
    self.p = np.clip(p, 0.0, 1.0)
    self._a = alpha
    
  def get_a(self):
    if len(self._a) == 2:
      return random.uniform(*self._a)
    else:
      return self._a
      
  def __call__(self, image):
    if self.p < random.random():
      return image
    _a = self.get_a()
    return image.filter(ImageFilter.UnsharpMask(_a))
    
  def __repr__(self):
    return '[Custom] ' + self.__class__.__name__ + f'(p={self.p}, alpha={self._a})'


class FilterMedian(object):
  def __init__(self, p=1.0, alpha=(1, 2)):
    self.p = np.clip(p, 0.0, 1.0)
    self._a = alpha
    
  def get_a(self):
    if len(self._a) == 2:
      return 2 * random.randint(*self._a) - 1
    else:
      return 2 * self._a - 1
      
  def __call__(self, image):
    if self.p < random.random():
      return image
    _a = self.get_a()
    return image.filter(ImageFilter.MedianFilter(_a))
    
  def __repr__(self):
    return '[Custom] ' + self.__class__.__name__ + f'(p={self.p}, alpha={self._a})'


class Distort(object):
  def __init__(self, p=1.0, r=128):
    self.p = p
    self.r = r
    
  def get_mesh(self, size):
    mesh_x = (size[0] // self.r) + 2
    mesh_y = (size[1] // self.r) + 2
    
    a0 = random.uniform(4, 16)
    a1 = random.uniform(4, 16)
    p0 = random.uniform(2, 10)
    p1 = random.uniform(2, 10)
    o0 = random.uniform(0, math.pi * 2 / p0)
    o1 = random.uniform(0, math.pi * 2 / p1)
    
    warp = [
            [
             (
                 np.clip(math.sin((j * self.r + o0) * p0) * a0 + i * self.r, 0, size[0]),
              np.clip(math.sin((i * self.r + o1) * p1) * a1 + j * self.r, 0, size[1]),
              )
             for j in range(mesh_y)
             ] for i in range(mesh_x)
             ]
             
    mesh = [
            (
                (i * self.r, j * self.r, (i + 1) * self.r, (j + 1) * self.r),
             (
                 warp[i][j][0], warp[i][j][1],
              warp[i][j + 1][0], warp[i][j + 1][1],
              warp[i + 1][j + 1][0], warp[i + 1][j + 1][1],
                    warp[i + 1][j][0], warp[i + 1][j][1],
                ),
            )
            for j in range(mesh_y - 1)
            for i in range(mesh_x - 1)
        ]
    return mesh
    
  def __call__(self, image, label):
    if self.p < random.random():
      return image, label
    mesh = self.get_mesh(image.size)
    image = image.transform(image.size, Image.MESH, mesh, Image.BILINEAR)
    label = label.transform(label.size, Image.MESH, mesh, Image.NEAREST)
    return image, label
  
  def __repr__(self):
    return '[Custom] ' + self.__class__.__name__ + f'(p={self.p}, r={self.r})'


class HairC2(object):
  def __init__(self, p=1.0):
    self.p = np.clip(p, 0.0, 1.0)
    
  def __call__(self, image):
    if self.p < random.random():
      return image
    hair_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    draw = ImageDraw.Draw(image)
    bx = random.randint(50, 400)
    by = int(bx * np.random.uniform(0.9, 1.1))
    w, h = image.size
    a0 = random.randint(0, 360)
    l0 = random.randint(50, 100)
    
    for i in range(80):
      x1 = random.randint(-int(0.3 * w), int(1.3 * w))
      y1 = random.randint(-int(0.3 * h), int(1.3 * h))
      
      x2 = x1 + bx + random.randint(-5, 25)
      y2 = y1 + by + random.randint(-5, 25)
      
      a1 = a0 + random.randint(-20, 20)
      a2 = a1 + l0 + random.randint(-20, 20)
      
      draw.arc((x1, y1, x2, y2), start=a1, end=a2, fill=hair_color, width=1)
      
      return image
      
  def __repr__(self):
    return '[Custom] ' + self.__class__.__name__ + f'(p={self.p})'


class HairS2(object):
  def __init__(self, p=1.0):
    self.p = np.clip(p, 0.0, 1.0)
    
  def __call__(self, image):
    if self.p < random.random():
      return image
    hair_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    draw = ImageDraw.Draw(image)
    a0 = 2 * np.pi * np.random.uniform()
    l0 = random.randint(20, 100)
    w, h = image.size
    amount = int(2000 / l0)
    
    for i in range(amount):
      xc = random.randint(-int(0.1 * w), int(1.1 * w))
      yc = random.randint(-int(0.1 * h), int(1.1 * h))
      
      ax = a0 + np.random.uniform(-0.3, 0.3)
      ay = a0 + np.random.uniform(-0.3, 0.3)
      
      l = l0 * np.random.uniform(0.9, 1.1)
      
      vx, vy = np.cos(ax), np.sin(ay)
      
      lx = int(l * vx)
      ly = int(l * vy)
      
      x1 = xc - lx
      y1 = yc - ly
      
      x2 = xc + lx
      y2 = yc + ly
      
      draw.line((x1, y1, x2, y2), fill=hair_color, width=1)
      
      return image
      
  def __repr__(self):
    return '[Custom] ' + self.__class__.__name__ + f'(p={self.p})'