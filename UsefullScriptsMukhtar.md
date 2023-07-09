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

install yaml if required:
pip install pyyaml

if having problems with cudnn checkout:
https://github.com/PaddlePaddle/PaddleDetection/issues/7629


## prepare hk dataset for recognition
python3 ./tools/prepare.py 

python3 tools/train.py -c configs/rec/PP-OCRv3/kz_PP-OCRv3_rec.yml -o Global.pretrained_model=./pretrain_models/en_PP-OCRv3_rec_train/best_accuracy

python3 tools/train.py -c configs/rec/PP-OCRv3/kz_PP-OCRv3_rec.yml -o Global.pretrained_model=./output/v3_kz_mobile/latest


## train from saved model
python3 tools/train.py -c configs/det/det_mv3_db.yml  \
-o Global.pretrained_model=./pretrain_models/MobileNetV3_large_x0_5_pretrained


## inference

Docs:
https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/inference_en.md

### -c Set the training algorithm yml configuration file
### -o Set optional parameters
### Global.pretrained_model parameter Set the training model address to be converted without adding the file suffix .pdmodel, .pdopt or .pdparams.
### Global.save_inference_dir Set the address where the converted model will be saved.

python3 tools/export_model.py -c ./models/v3_kz_mobile/config.yml -o Global.pretrained_model=./models/v3_kz_mobile/best_accuracy  Global.save_inference_dir=./inference/svtr_kz/