## prepare data for training and testing

Note: fix scripts to match simple_dataset.py which reads base dir from yaml config

python ./ppocr/utils/gen_label.py --mode="det" --root_path="./train_data/icdar2015/text_localization/ch4_training_images/"  \
--input_path="./train_data/icdar2015/text_localization/ch4_training_localization_transcription_gt" \
--output_label="./train_data/icdar2015/text_localization/train_icdar2015_label.txt"

python ./ppocr/utils/gen_label.py --mode="det" --root_path="./train_data/icdar2015/text_localization/ch4_test_images/"  \
--input_path="./train_data/icdar2015/text_localization/Challenge4_Test_Task1_GT" \
--output_label="./train_data/icdar2015/text_localization/test_icdar2015_label.txt"

##  prepare data for training rec model
python ./ppocr/utils/gen_label.py --mode="rec" \
--input_path="./train_data/ic15_data/train/gt.txt" \
--output_label="./train_data/ic15_data/rec_gt_train.txt"

python ./ppocr/utils/gen_label.py --mode="rec"  --input_path="./train_data/ic15_data/test/gt.txt" --output_label="./train_data/ic15_data/rec_gt_test.txt"

## train det
python3 tools/train.py -c configs/det/det_mv3_db.yml  \
-o Global.pretrained_model=./pretrain_models/MobileNetV3_large_x0_5_pretrained

## train rec
python3 tools/train.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.pretrained_model=./pretrain_models/en_PP-OCRv3_rec_train/best_accuracy


# repository setup with CUDA

pip install -r requirements.txt

install yaml if required:
pip install pyyaml

if having problems with cudnn checkout:
https://github.com/PaddlePaddle/PaddleDetection/issues/7629

https://developer.nvidia.com/compute/cudnn/secure/8.4.1/local_installers/11.6/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz

sudo unlink libcudnn.so
sudo unlink libcublas.so

locate libcudnn.so
locate libcublas.so

sudo ln -s /home/bma/miniforge3/pkgs/cudnn-8.4.1.50-hed8a83a_0/lib/libcudnn.so.8.4.1 libcudnn.so
sudo ln -s /home/bma/miniforge3/pkgs/cudatoolkit-11.7.1-h4bc3d14_12/lib/libcublas.so.11.10.3.66 libcublas.so

ls /usr/lib | grep lib


## prepare hk dataset for recognition
python3 ./tools/prepare.py 

python3 tools/train.py -c configs/rec/PP-OCRv3/kz_PP-OCRv3_rec.yml -o Global.pretrained_model=./pretrain_models/en_PP-OCRv3_rec_train/best_accuracy

python3 tools/train.py -c configs/rec/PP-OCRv3/kz_PP-OCRv3_rec.yml -o Global.pretrained_model=./output/v3_kz_mobile/latest


## train from saved model
python3 tools/train.py -c configs/det/det_mv3_db.yml  \
-o Global.pretrained_model=./pretrain_models/MobileNetV3_large_x0_5_pretrained

## train on synthetic dataset
python3 tools/train.py -c configs/rec/PP-OCRv3/kz_synthetic_PP-OCRv3_rec.yml -o Global.pretrained_model=./output/v3_kz_mobile_synthetic_hand/latest
python3 tools/train.py -c configs/rec/PP-OCRv3/kz_synthetic_PP-OCRv3_rec.yml -o Global.pretrained_model=./output/v3_kz_mobile_synthetic_multi_word/latest
python3 tools/train.py -c configs/rec/PP-OCRv3/kz_synthetic_PP-OCRv3_rec.yml -o Global.pretrained_model=./output/v3_kz_mobile/latest
python3 tools/train.py -c configs/rec/PP-OCRv3/kz_synthtiger_rec.yml -o Global.pretrained_model=./output/kz_synthtiger_rec/latest

wsl
cd
cd PaddleOCR_Mukhtar
conda activate paddle_env
python3 tools/train.py -c configs/rec/PP-OCRv4/kz_PP-OCRv4_rec_v1.yml

## inference

Docs:
https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/inference_en.md

### -c Set the training algorithm yml configuration file
### -o Set optional parameters
### Global.pretrained_model parameter Set the training model address to be converted without adding the file suffix .pdmodel, .pdopt or .pdparams.
### Global.save_inference_dir Set the address where the converted model will be saved.

python3 tools/export_model.py -c ./output/v3_kz_mobile_synthetic/config.yml -o Global.pretrained_model=./output/v3_kz_mobile_synthetic/best_accuracy  Global.save_inference_dir=./inference/v3_kz_mobile_synthetic_test/
python3 tools/export_model.py -c ./output/kz_synthtiger_rec_v6/config.yml -o Global.pretrained_model=./output/kz_synthtiger_rec_v6/best_accuracy  Global.save_inference_dir=./inference/kz_synthtiger_rec/
python3 tools/export_model.py -c ./output/ppocr3_kz_synthtiger/config.yml -o Global.pretrained_model=./output/ppocr3_kz_synthtiger/best_accuracy  Global.save_inference_dir=./inference/ppocr3_kz_synthtiger/


## deployment
pip install paddlelite==2.10
paddle_lite_opt

# deploy

android
https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_en/android_demo_en.md


# data generation
https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/data_synthesis_en.md

# 【[Recommendation] Download the Chinese and English inference model of PP-OCRv3
wget  https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_slim_infer.tar && tar xf  ch_PP-OCRv3_det_slim_infer.tar
wget  https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.tar && tar xf  ch_PP-OCRv2_rec_slim_quant_infer.tar
wget  https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_cls_slim_infer.tar && tar xf  ch_ppocr_mobile_v2.0_cls_slim_infer.tar
# Convert detection model
paddle_lite_opt --model_file=./ch_PP-OCRv3_det_slim_infer/inference.pdmodel  --param_file=./ch_PP-OCRv3_det_slim_infer/inference.pdiparams  --optimize_out=./ch_PP-OCRv3_det_slim_opt --valid_targets=arm  --optimize_out_type=naive_buffer
# Convert recognition model
paddle_lite_opt --model_file=./ch_PP-OCRv3_rec_slim_infer/inference.pdmodel  --param_file=./ch_PP-OCRv3_rec_slim_infer/inference.pdiparams  --optimize_out=./ch_PP-OCRv3_rec_slim_opt --valid_targets=arm  --optimize_out_type=naive_buffer
# Convert angle classifier model
paddle_lite_opt --model_file=./ch_ppocr_mobile_v2.0_cls_slim_infer/inference.pdmodel  --param_file=./ch_ppocr_mobile_v2.0_cls_slim_infer/inference.pdiparams  --optimize_out=./ch_ppocr_mobile_v2.0_cls_slim_opt --valid_targets=arm  --optimize_out_type=naive_buffer

paddle_lite_opt --model_file=./inference/svtr_kz_hand/inference.pdmodel --param_file=./inference/svtr_kz_hand/inference.pdiparams --optimize_out=./inference/svtr_kz_hand_lite --valid_targets=arm  --optimize_out_type=naive_buffer
