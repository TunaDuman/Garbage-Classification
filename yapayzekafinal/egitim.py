import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom, Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

VERI_YOLU = "C:/Users/tunad/Desktop/yapayzekafinal/Garbageclassification" #burada kendi dosya yolunuz yazılmazsa model eğitilemez.
BOYUT = (224, 224)
BATCH_SIZE = 32

print("Veriler yükleniyor")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    VERI_YOLU, validation_split=0.2, subset="training", seed=123,
    image_size=BOYUT, batch_size=BATCH_SIZE, label_mode='categorical'
)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VERI_YOLU, validation_split=0.2, subset="validation", seed=123,
    image_size=BOYUT, batch_size=BATCH_SIZE, label_mode='categorical'
)

class_names = train_dataset.class_names

data_augmentation = tf.keras.Sequential([
  RandomFlip("horizontal_and_vertical"), 
  RandomRotation(0.3),                  
  RandomZoom(0.1),                      
])

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)

x = Rescaling(1./127.5, offset=-1)(x)

x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

erken_durdurma = EarlyStopping(
    monitor='val_accuracy', 
    patience=3,             
    restore_best_weights=True 
)

print("eğitim başlıyor")

history = model.fit(
    train_dataset, 
    validation_data=validation_dataset, 
    epochs=30, 
    callbacks=[erken_durdurma]
)

model.save('ayiricimodel.h5')
print("Model başarıyla eğitildi ve 'ayiricimodel.h5' olarak kaydedildi.")