import glob
import os
import shutil

from keras.callbacks import ModelCheckpoint

from config import BATCH_SIZE, CONFIGS, DATA_GEN_ARGS, EPOCHS
from src.generator import customGenerator
from src.processing import preprocessing
from src.util import cleanup, folder_check, get_uniquename, get_latestname
from src.models import load_resnet50

def main(prepare, train):
    categories = os.listdir('images/train')
    print("Detected classes:", categories)
    
    if prepare:
        print("Create Dataset")
        cleanup()
        folder_check()

        create_dataset(categories)

    if train:
        print("Run training")
        run_training()
        pass


def create_dataset(categories):
    os.makedirs('__dataset__')

    for category in categories:
        for datatype in ['train', 'valid']:
            p_images = glob.glob(f'images/{datatype}/{category}/*')
            os.makedirs(f'__dataset__/{datatype}/{category}')
            outdir = f'__dataset__/{datatype}/{category}'
            for p_image in p_images:
                preprocessing(p_image, outdir)


def run_training():
    trainGene = customGenerator(batch_size=BATCH_SIZE,
                                train_path='__dataset__',
                                image_folder='train',
                                aug_dict=DATA_GEN_ARGS,
                                save_to_dir=None,
                                image_color_mode="rgb",)

    validGene = customGenerator(batch_size=BATCH_SIZE,
                                train_path='__dataset__',
                                image_folder='valid',
                                aug_dict=None,
                                save_to_dir=None,
                                image_color_mode="rgb",)

    if os.path.exists("__checkpoints__"):
        shutil.rmtree("__checkpoints__")
    os.makedirs("checkpoints")
     
    hdfname = get_uniquename("__checkpoints__/model_", 1)
    model_checkpoint = ModelCheckpoint('{}.hdf5'.format(hdfname), 
                                       monitor='loss', verbose=1,
                                       save_best_only=True)

    n_classes = len(os.listdir('__dataset__/train/*'))
    n_train_images = len(glob.glob('__dataset__/train/*/*'))
    n_valid_images = len(glob.glob('__dataset__/valid/*/*'))

    trained_weight = get_latestname("__checkpoints__/model_", 1) 
    model = load_resnet50(n_classes, trained_weight) 
    model.fit_generator(trainGene, 
                        steps_per_epoch=n_train_images // BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=validGene,
                        validation_steps=n_valid_images // BATCH_SIZE,
                        callbacks=[model_checkpoint])


if __name__ == '__main__':
    main(prepare=False, train=True)
