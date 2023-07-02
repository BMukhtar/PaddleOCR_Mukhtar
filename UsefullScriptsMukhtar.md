
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

## train
python3 tools/train.py -c configs/det/det_mv3_db.yml  \
-o Global.pretrained_model=./pretrain_models/MobileNetV3_large_x0_5_pretrained


## train from saved model
python3 tools/train.py -c configs/det/det_mv3_db.yml  \
-o Global.pretrained_model=./pretrain_models/MobileNetV3_large_x0_5_pretrained
