from utils import load_dataset
import ann_models
from tensorflow.keras.models import save_model
from datetime import datetime
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

DATASET_PATH = "./datasets/lfsr_11-9_random.h5"

# load dataset
train_data, validation_data = load_dataset(DATASET_PATH, 0.25)
(x_train, y_train) = train_data

# generate network types
(model, model_name) = ann_models.get_fully_connected_model(
    input_shape=x_train.shape[1:], data_name="lfsr_11-9")

# compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["binary_accuracy"]
)

# train the model
results = model.fit(
    x=x_train,
    y=y_train,
    validation_data=validation_data,
    epochs=5,
    batch_size=50
    #   callbacks=[tensorboard],
)

# save the model
# save_model(
#     model,
#     f'./models/{model_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.model.h5',
#     overwrite=True,
#     include_optimizer=True,
# )


(x_test, y_test) = validation_data

predictions = model.predict(
    x_test,
    batch_size=None,
    verbose=0,
    steps=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False
)

# false-positive-rate, true-positive-rate, thresholds
fpr_model, tpr_model, thresholds_model = roc_curve(y_test, predictions)

auc_keras = auc(fpr_model, tpr_model)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_model, tpr_model, label='Model (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
