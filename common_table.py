# -*- coding: utf-8 -*-

import warnings, os, time
warnings.filterwarnings("ignore")
from loguru import logger as log
if not os.path.exists('./log'):
    os.makedirs('./log')
# log.remove()  # 删去import logger之后自动产生的handler，不删除的话会出现重复输出的现象
log.add(os.path.join('./log', f'{time.strftime("%Y_%m_%d")}.log'), retention="10 days")


from ocr_system_base import load_model, OCR
from common.ocr_utils import string_to_arrimg
from common.exceptions import ParsingError
from common.params import args
import table_ceil
from common.timeit import time_it

e2e_algorithm, text_sys = load_model(args, e2e_algorithm = False)
# log.info(args.__dict__)



@time_it
def main(img_str, isTransform=True, perspectiveTransform=True,
         full_ocr=True, cell_recognition=False, save_path = './test', img_prefix = 'result'):
    if isinstance(img_str, str):
        img = string_to_arrimg(img_str, log_flag = True)
        if img is None:
            raise ParsingError('Failed to transform base64 to image.', 2)
    else:
        img = img_str


    if full_ocr and cell_recognition==False:
        ocr = OCR(text_sys, img, cls = False, char_out = True, e2e_algorithm = False)
    else:
        ocr = None
    tableRec = table_ceil.Table(img, text_sys, ocr, isTableDetect=True,
                                isTransform=isTransform, perspectiveTransform=perspectiveTransform,
                                isImgProcess=False, cell_recognition = cell_recognition, save_path = save_path, img_prefix = img_prefix)
    #_, tableJson = tableRec(char_out = True)
    _, tableJson = tableRec()
    return tableJson


if __name__ == "__main__":
    import os, json
    from common.ocr_utils import imagefile_to_string

    filepath = 'test/16'
    #filepath = '914_test1_youxianbiaoge'
    n = 0
    for maindir, subdir, file_name_list in list(os.walk(filepath)):
        for file in file_name_list:
            if file.rsplit('.')[-1].lower() in ['bmp', 'png', 'jpg', 'jpeg'] and \
                'ceil' not in file and 'seq' not in file:
                n += 1
                apath = os.path.join(maindir, file)  # 合并成一个完整路径
                apath = r'test/16/20240423_1.png'
                print('处理第[%d]个文件[%s]' % (n, apath))
                img_str = imagefile_to_string(apath)
                save_path, img_prefix = os.path.dirname(apath), os.path.basename(apath).split('.')[0]
                tableJson = main(img_str,
                                 isTransform=True, perspectiveTransform=True,
                                 full_ocr = True, save_path=save_path, img_prefix=img_prefix)
                with open(apath.rsplit('.', 1)[0] + '.json', 'w', encoding = 'utf8') as f:
                    f.write(json.dumps(tableJson, ensure_ascii = False, sort_keys = False, separators = (",", ":")))
                # print(tableJson)
                break
        break
    print('-------END-------')
