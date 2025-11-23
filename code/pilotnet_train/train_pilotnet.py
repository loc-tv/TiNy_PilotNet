# train_pilotnet.py

import tensorflow as tf
import numpy as np
from model_pilotnet import PilotNetSmall
from dataset_pilotnet import PilotNetConcatDatasetTF
import config_train as cfg


def dataset_to_tf(ds):
    output_types = (tf.float32, tf.float32)
    output_shapes = ((120, 160, 1), ())

    tf_ds = tf.data.Dataset.from_generator(
        ds.generator,
        output_types=output_types,
        output_shapes=output_shapes
    )

    return tf_ds


def main():
    print("="*50)
    print("🚀 PILOTNET KERAS TRAINING START")
    print("="*50)

    datasets = PilotNetConcatDatasetTF.load_from_config(cfg.DATASETS)

    # Gom dataset lại
    full_tf_datasets = [dataset_to_tf(ds) for ds in datasets]
    full_dataset = full_tf_datasets[0]
    for ds in full_tf_datasets[1:]:
        full_dataset = full_dataset.concatenate(ds)

    full_dataset = full_dataset.shuffle(5000)

    total_len = sum([len(ds) for ds in datasets])
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    # Split giống PyTorch
    # train_ds = full_dataset.take(train_len).batch(cfg.BATCH_SIZE)
    # val_ds = full_dataset.skip(train_len).take(val_len).batch(cfg.BATCH_SIZE)
    # test_ds = full_dataset.skip(train_len + val_len).batch(cfg.BATCH_SIZE)
    train_ds = (
        full_dataset
        .take(train_len)
        .shuffle(5000)
        .batch(cfg.BATCH_SIZE)
        .repeat()                      
        .prefetch(tf.data.AUTOTUNE)
    )
    
    val_ds = (
        full_dataset
        .skip(train_len)
        .take(val_len)
        .batch(cfg.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        full_dataset
        .skip(train_len + val_len)
        .batch(cfg.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


    print(f"[DATASET SPLIT]")
    print(f"Train: {train_len}")
    print(f"Val:   {val_len}")
    print(f"Test:  {test_len}\n")

    # Model
    model = PilotNetSmall()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.LR),
        loss='mse',
        metrics=['mae']
    )
    
    steps_per_epoch = train_len // cfg.BATCH_SIZE
    validation_steps = val_len // cfg.BATCH_SIZE


    # model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     epochs=cfg.EPOCHS
    # )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.EPOCHS,
        steps_per_epoch=steps_per_epoch,          
        validation_steps=validation_steps         
    )


    # Evaluate
    test_loss, test_mae = model.evaluate(test_ds)
    print(f"\n🎯 Final Test Loss: {test_loss:.6f}")

    # Save model
    model.save(cfg.MODEL_OUT)
    print(f"✅ Saved model to: {cfg.MODEL_OUT}")


if __name__ == "__main__":
    main()
