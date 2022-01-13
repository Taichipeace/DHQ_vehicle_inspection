"""
#导出keypoints模型
python tools/export_model.py -c configs/keypoint/hrnet/dark_hrnet_w32_256x192.yml -o weights=D:/MVproject/ceshi/00TedOCR/TedMod/keypoints/dark_hrnet_w32_256x192.pdparams

#导出FairMOT跟踪模型
python tools/export_model.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=D:/MVproject/ceshi/00TedOCR/TedMod/keypoints/fairmot_dla34_30e_1088x608.pdparams

#用导出的跟踪和关键点模型Python联合预测
python deploy/python/mot_keypoint_unite_infer.py --mot_model_dir=D:/MVproject/ceshi/00TedOCR/TedMod/keypoints/fairmot_dla34_30e_1088x608/ --keypoint_model_dir=D:/MVproject/ceshi/00TedOCR/TedMod/keypoints/dark_hrnet_w32_256x192/ --video_file=D:/MVproject/ceshi/00TedOCR/FunBasketball/296967120-1-208small.mp4 --device=GPU --save_mot_txts --output_dir=D:/MVproject/ceshi/00TedOCR/FunBasketball/Output/
"""
import paddle
import mot_keypoint_unite_infer as uifer
from infer import print_arguments, PredictConfig
from mot_jde_infer import JDE_Detector
from keypoint_infer import KeyPoint_Detector, PredictConfig_KeyPoint
from mot_keypoint_unite_utils import argsparser


paddle.enable_static()
parser = argsparser()
FLAGS = parser.parse_args()
FLAGS.video_file = 'D:/MVproject/ceshi/00TedOCR/FunBasketball/296967120-1-208small.mp4'
FLAGS.save_mot_txts = True
FLAGS.output_dir = 'D:/MVproject/ceshi/00TedOCR/FunBasketball/Output/'
FLAGS.mot_model_dir = 'D:/MVproject/ceshi/00TedOCR/TedMod/keypoints/fairmot_dla34_30e_1088x608/'
FLAGS.keypoint_model_dir = 'D:/MVproject/ceshi/00TedOCR/TedMod/keypoints/dark_hrnet_w32_256x192/'
FLAGS.device = 'GPU'
print(type(FLAGS))
print_arguments(FLAGS)

# mot JDE 模型初始化
pred_config = PredictConfig(FLAGS.mot_model_dir)
mot_model = JDE_Detector(
        pred_config,
        FLAGS.mot_model_dir
        )

# keypoint 模型初始化
pred_config = PredictConfig_KeyPoint(FLAGS.keypoint_model_dir)
keypoint_model = KeyPoint_Detector(
        pred_config,
        FLAGS.keypoint_model_dir
        )

# 分析视频并保存txt结果
uifer.mot_keypoint_unite_predict_video(FLAGS,
                                       mot_model,
                                       keypoint_model,
                                       camera_id=-1,
                                       keypoint_batch_size=1
                                       )
