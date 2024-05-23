# -*- coding: utf-8 -*-

import os

from common.params import args

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
cpu = args.use_gpu == False
if cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

#一旦不再使用即释放内存垃圾，=1.0 垃圾占用内存大小达到10G时，释放内存垃圾
os.environ["FLAGS_eager_delete_tensor_gb"]="0.0"
#启用快速垃圾回收策略，不等待cuda kernel 结束，直接释放显存
os.environ["FLAGS_fast_eager_deletion_mode"]="1"
#该环境变量设置只占用0%的显存
os.environ["FLAGS_fraction_of_gpu_memory_to_use"]="0"

import cv2
import numpy as np
from PIL import Image
from loguru import logger as log
from ppocr.infer.predict_system import TextSystem
from ppocr.infer.predict_e2e import TextE2E
from common.ocr_utils import string_to_arrimg, order_points, fourxy2twoxy
from common.exceptions import ParsingError
from common.box_util import stitch_boxes_into_lines_v2 as stitch_boxes_into_lines
from ppocr.infer.utility import draw_ocr_box_txt, draw_e2e_res
import copy, paddle


def load_model(args, e2e_algorithm=False):
    log.info("Loading model...")
    if args.use_gpu:
        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
            log.info("use gpu: %s"%args.use_gpu)
            log.info("CUDA_VISIBLE_DEVICES: %s"%_places)
            args.gpu_mem = 500
        except:
            raise RuntimeError(
                "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
            )
    else:
        log.info("use gpu: %s"%args.use_gpu)



    if e2e_algorithm:
        text_sys = TextE2E(args)
    else:
        text_sys = TextSystem(args)
    # log.info(args.__dict__)
    if args.warmup:
        img_warm = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(10):
            _ = text_sys(img_warm)
    return e2e_algorithm, text_sys


class OCR():
    def __init__(self, text_sys, img_str, cls = False, char_out = False, e2e_algorithm=False):
        if isinstance(img_str,str):
            img = string_to_arrimg(img_str, log_flag = True)
            if img is None:
                raise ParsingError('Failed to transform base64 to image.', 4)
        else:
            img = img_str

        self.img = img
        self.char_out = char_out
        self.e2e_algorithm = e2e_algorithm


        if self.e2e_algorithm:
            self.dt_boxes, self.rec_res = text_sys(img)
        else:
            self.dt_boxes, self.rec_res = text_sys(img, cls, char_out)

    def __call__(self, union, max_x_dist = 50, min_y_overlap_ratio = 0.5):
        img = self.img
        img_origin = copy.deepcopy(img)
        char_out = self.char_out

        if self.e2e_algorithm:
            dt_boxes, rec_res = self.dt_boxes, self.rec_res
            dt_num = len(dt_boxes)
            result = []
            for dno in range(dt_num):
                result.append({"text": rec_res[dno], "box": (dt_boxes[dno]).tolist()})

            if args.is_visualize:
                src_im = draw_e2e_res(dt_boxes, rec_res, img)
                cv2.imwrite('./test/draw_img.jpg', src_im)
                log.info("The visualized image saved in ./test/draw_img.jpg")
        else:
            dt_boxes, rec_res = self.dt_boxes, self.rec_res
            dt_num = len(dt_boxes)
            result = []
            for dno in range(dt_num):
                text, score, chars_list = rec_res[dno]
                if chars_list is None:
                    chars_list = list()
                if len(text) == 1:
                    chars_list = [dt_boxes[dno].astype('int').tolist()]
                else:
                    chars_list = [order_points(np.array(i)).astype('int').tolist() for i in chars_list]

                chars = list()
                for c, p in enumerate(chars_list):
                    chars.append({
                        "text": text[c],
                        # "quadrangle": p,
                        # "box": np.array(p).reshape(1, -1).squeeze().tolist(),
                        "bbox": fourxy2twoxy(p)
                    })
                quadrangle = dt_boxes[dno]
                temp_result = {"text": text,
                               "quadrangle": quadrangle.tolist(),
                               "box": quadrangle.reshape(1, -1).squeeze().tolist(),
                               "bbox": fourxy2twoxy(quadrangle),
                               "score": float(score)}
                if char_out:
                    temp_result.update({"chars": chars})
                result.append(temp_result)

                if char_out and args.is_visualize_char:
                    if chars_list is not None:
                        for ps in chars_list:
                            # cv2.line(img, tuple(ps[0]), tuple(ps[3]), (0, 0, 255), 2, 4)
                            cv2.polylines(img_origin, [np.array(ps, dtype = 'int32')],
                                          isClosed = True, color = (255, 255, 0), thickness = 1)

            if union:
                result = stitch_boxes_into_lines(result, max_x_dist = max_x_dist,
                                                       min_y_overlap_ratio = min_y_overlap_ratio)

            if args.is_visualize:
                image = Image.fromarray(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB))
                if union:
                    boxes = [np.array(i['quadrangle']) for i in result]
                    txts = [i['text'] for i in result]
                    scores = [i['score'] for i in result]
                else:
                    boxes = dt_boxes
                    txts = [rec_res[i][0] for i in range(len(rec_res))]
                    scores = [rec_res[i][1] for i in range(len(rec_res))]

                draw_img = draw_ocr_box_txt(
                    image,
                    boxes,
                    txts,
                    scores,
                    drop_score=args.drop_score,
                    font_path=args.vis_font_path)
                cv2.imwrite('./test/draw_img.jpg', draw_img[:, :, ::-1])
                log.info("The visualized image saved in ./test/draw_img.jpg")
        paddle.device.cuda.empty_cache()
        return result


if __name__ == "__main__":
    import cv2, json
    from common.ocr_utils import imagefile_to_string

    e2e_algorithm, text_sys = load_model(args, e2e_algorithm = False)
    # log.info(args.__dict__)

    filename = r'test/16/6.jpg'
    img = imagefile_to_string(filename)
    # print(type(img))

    # img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
    # print(type(img))
    ocr = OCR(text_sys, img, cls = False, char_out = True, e2e_algorithm = e2e_algorithm)
    result = ocr(union = True, max_x_dist = 1000, min_y_overlap_ratio = 0.5)

    # with open(filename.rsplit('.', 1)[0] + '.json', 'w', encoding='utf8') as f:
    #     f.write(json.dumps(result, ensure_ascii=False, sort_keys=False, separators=(",", ":")))
    print(result)

    # 键值对信息抽取
    #for i in result:
        #print(i['text'])