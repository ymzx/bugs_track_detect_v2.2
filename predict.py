import os, paddle
from paddleseg.cvlibs import Config
from paddleseg.utils import get_sys_env, logger, config_check
from paddleseg.core import predict
from configs.user_cfg import model_path, cfg_path, image_path, save_dir


def get_image_list(image_path):
    """Get image list"""
    valid_suffix = [
        '.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png'
    ]
    image_list = []
    image_dir = None
    if os.path.isfile(image_path):
        if os.path.splitext(image_path)[-1] in valid_suffix:
            image_list.append(image_path)
        else:
            image_dir = os.path.dirname(image_path)
            with open(image_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line.split()) > 1:
                        line = line.split()[0]
                    image_list.append(os.path.join(image_dir, line))
    elif os.path.isdir(image_path):
        image_dir = image_path
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if '.ipynb_checkpoints' in root:
                    continue
                if os.path.splitext(f)[-1] in valid_suffix:
                    image_list.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(
            '`--image_path` is not found. it should be an image file or a directory including images'
        )

    if len(image_list) == 0:
        raise RuntimeError('There are not image file in `--image_path`')

    return image_list, image_dir


def main():
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info['GPUs used'] else 'cpu'

    paddle.set_device(place)

    cfg = Config(cfg_path)
    val_dataset = cfg.val_dataset
    if not val_dataset:
        raise RuntimeError('The verification dataset is not specified in the configuration file.')

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model # BisenetV2
    transforms = val_dataset.transforms
    image_list, image_dir = get_image_list(image_path)
    logger.info('Number of predict images = {}'.format(len(image_list)))

    config_check(cfg, val_dataset=val_dataset)

    predict(
        model,
        model_path=model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=save_dir)


if __name__ == '__main__':
    main()
