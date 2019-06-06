import os
import pathlib

from PIL import Image


def preprocessing(p_image, outdir=None):
    name = pathlib.Path(p_image).stem
    image = Image.open(p_image)

    images, names = modification(image, name)

    if outdir:
        for image, name in zip(images, names):
            image.save(os.path.join(outdir, name+'.jpg'))

    return images, names


def modification(image, name):
    """ 画像加工用の関数
        例えばリサイズとか一枚の画像を４分割するとか、ぼかしをかけるとか
        ブートストラップで画像水増しとか,そういう処理を記述する
        デフォルトでは推論時もこの関数が呼ばれるので、とくに
        画像水増しなんかするなら整合性に注意
    """
    images = [image]
    names = [name]

    images = [image.resize((512,512)) for image in images]
    names = [name for name in names]

    return images, names


def postprocessing(image, result):
    return image, result
