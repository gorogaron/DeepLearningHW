import keras.backend as K
K.set_image_dim_ordering("th")

from keras.optimizers import SGD, adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from networks import UNET
from imageGenerator import train_generator, valid_generator

NB_EPOCHS = 1
BATCH_SIZE= 1

optimizer = adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
loss      = "mse"

model = UNET((6,128,384))
model.compile(loss=loss, optimizer=optimizer)

callbacks = [
        ModelCheckpoint(filepath="./model_weights/weights.hdf5", monitor='loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="loss", factor=0.5, patience=20, verbose=1)
]

hist = model.fit_generator(
    generator=train_generator(BATCH_SIZE),
    steps_per_epoch = 2,
    validation_data=valid_generator(),
    validation_steps= 2,
    epochs = NB_EPOCHS,
    callbacks=callbacks
)