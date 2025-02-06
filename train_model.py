import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from preprocess import train_dataset, val_dataset, test_dataset

# Number of classes (23 skin conditions)
NUM_CLASSES = 23

# Load EfficientNetB0 with pretrained ImageNet weights
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)  # Add dropout to prevent overfitting
x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(NUM_CLASSES, activation='softmax')(x)

# Create model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model (First Phase - Feature Extraction)
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,  # Initial training
    callbacks=[early_stopping, checkpoint]
)

# After initial training, we unfreeze some layers for fine-tuning
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Recompile with a lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.00005), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Fine-tune model (Second Phase)
history_finetune = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,  # Fine-tuning
    callbacks=[early_stopping, checkpoint]
)

# Save final model
model.save('efficientnetb0_skin_conditions.h5')

print("Training and fine-tuning complete!")
