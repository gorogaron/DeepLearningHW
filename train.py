import keras.backend as K
from keras.optimizers import SGD, adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from networks import UNET
from losses import charbonnier, soft_dice
from imageGenerator import train_generator, valid_generator
K.set_image_dim_ordering("th")

NB_EPOCHS = 60
BATCH_SIZE= 1

optimizer = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
loss      = soft_dice

model = UNET((6,128,384))
model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

callbacks = [
        ModelCheckpoint(filepath="./model_weights/weights_dice_loss.hdf5", monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
]

hist = model.fit_generator(
    generator=train_generator(BATCH_SIZE),
    steps_per_epoch = 6144,
    validation_data=valid_generator(BATCH_SIZE),
    validation_steps= 2048,
    epochs = NB_EPOCHS,
    callbacks=callbacks
)