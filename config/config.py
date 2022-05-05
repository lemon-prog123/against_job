import argparse
spectral=False
quant=False
regular=False
activate=False
regular_mode=1
save_path='checkpoint'
img_path='img'
iteration1=15
iteration2=10
squeeze=-1
norm=True
argparser = argparse.ArgumentParser(description='Against Task')
argparser.add_argument('--spectral', action='store_true',help='Sepctral Norm')
argparser.add_argument('--regular', action='store_true', help='Regular')
argparser.add_argument('--PGD_train', action='store_true', help='PGD_train')
argparser.add_argument('--quant', action='store_true',help='Quant Train')
argparser.add_argument('--activate', action='store_true',help='Activate Train')
argparser.add_argument('--checkpoint', type=str, help='Checkpoint to load')
argparser.add_argument('--savepath', type=str, help='Checkpoint save path')
argparser.add_argument('--imgpath', type=str, help='Image save path')
argparser.add_argument('--iterations', type=int, help='Epochs')
argparser.add_argument('--squeeze', type=int, help='Squeeze Train')
argparser.add_argument('--eval', action='store_true', help='Eval Mode')
argparser.add_argument('--denorm', action='store_true', help='No normdata')