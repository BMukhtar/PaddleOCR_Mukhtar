from paddleocr import PaddleOCR


def main():
    ocr = PaddleOCR(
        lang="ru",
        use_angle_cls="true",
        rec_algorithm='SVTR',
        rec_model_dir='./inference/kz_synthtiger_rec',
        rec_char_dict_path='./ppocr/utils/dict/kz_dict.txt',
        use_gpu=False,
    )  # need to run only once to download and load model into memory

    img_path = './train_data/test_100/IMG_6105.jpeg'
    print(ocr.ocr(img_path, rec=True))


if __name__ == "__main__":
    main()
