from keras.utils import plot_model
from keras.models import load_model

model = load_model('model2.model')
plot_model(model, to_file='model.png', show_shapes=True)