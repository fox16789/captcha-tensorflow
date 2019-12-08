from captcha.image import ImageCaptcha
import numpy as np 
from PIL import Image
import random
import sys

number = ['0','1','2','3','4','5','6','7','8','9']

def random_captcha_text(char_set=number, char_size=4):
    # 生成验证码数字
    captcha_text = []
    for i in range(char_size):
        c=random.choice(char_set)
        captcha_text.append(c)

    return captcha_text

# 生成验证码图片
def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha_text() # 列表
    captcha_text = ''.join(captcha_text) # 转为字符串
    captcha = image.generate(captcha_text) # 生成图片
    image.write(captcha_text,'TensorFlow/images/' + captcha_text + '.jpg') # 写入文件

num = 10000
if __name__ == "__main__":
    for i in range(num):
        gen_captcha_text_and_image()
        sys.stdout.write('\r>>Creating images %d/%d' %(i+1,num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()

    print("ok")