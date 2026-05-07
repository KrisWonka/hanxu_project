import cv2
from rknnlite.api import RKNNLite

try:
  from .coco_utils import COCO_test_helper
  from .model import Model
  from .rknn_model_postprocess import RknnModelPostProcessor
except ImportError:
  from coco_utils import COCO_test_helper
  from model import Model
  from rknn_model_postprocess import RknnModelPostProcessor

class RknnliteModel(Model):
  def __init__(self, model_path, img_size=(640, 640), obj_thresh=0.5, nms_thresh=0.45):
    self.rknn_lite = RKNNLite()
    self.img_size = img_size
    self.obj_thresh = obj_thresh
    self.nms_thresh = nms_thresh
    self.post_processor = RknnModelPostProcessor(self.img_size, self.nms_thresh, self.obj_thresh)
    
    ret = self.rknn_lite.load_rknn(model_path)
    if ret != 0:
      raise Exception(f'Load RKNN model failed from {model_path}')
      
    ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
      raise Exception('Init runtime env failed')
    
    self.co_helper = COCO_test_helper(enable_letter_box=True)

  def __del__(self):
    self.release()

  def release(self):
    if getattr(self, 'rknn_lite', None) is not None:
      self.rknn_lite.release()
      self.rknn_lite = None

  # 核心方法
  def detect(self, orig_img):
    img = self.co_helper.letter_box(
            im=orig_img.copy(), 
            new_shape=(self.img_size[1], self.img_size[0]), 
            pad_color=(0,0,0)
        )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    outputs = self.rknn_lite.inference(inputs=[img])
    boxes, classes, scores = self.post_processor.post_process(outputs)
        
    if boxes is not None:
        realBoxes = self.co_helper.get_real_box(boxes)
        # top, left, right, bottom 
        return [
            [float(box[0]), float(box[1]), float(box[2]), float(box[3]), 
              float(scores[i]), int(classes[i])] 
            for i, box in enumerate(realBoxes)
        ]
    return []
