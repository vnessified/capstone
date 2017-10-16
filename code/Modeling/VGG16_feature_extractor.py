import numpy as np
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

def VGG16_feature_maps(sample_count, directory, batch_size, class_list, shuffle):
    
    '''
    Uses VGG16 a feature extractor on image data for transfer learning.
    These feature maps are then passed through a model to generate image classes.
    
    # Arguments
        sample_count: number of data samples.
        directory: filepath to data - classes should be organized into subdirectories.
        batch_size: batch size of data supplied to ImageDataGenerator class.
        class_list: list specifies class encoding by list index.
        shuffle: boolean - shuffles the data.     
    '''
    
    
    vgg16_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

    datagen = ImageDataGenerator(rescale=1./255)

    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    
    if shuffle == True:
        generator = datagen.flow_from_directory(
            directory,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='binary',
            classes=class_list,
            shuffle=True)
    else:
        generator = datagen.flow_from_directory(
            directory,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='binary',
            classes=class_list,
            shuffle=False)
        
    print generator.class_indices
    
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = vgg16_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        # Note that since generators yield data indefinitely in a loop,
        # we must `break` after every image has been seen once.
        if i * batch_size >= sample_count:
            break
    return features, labels