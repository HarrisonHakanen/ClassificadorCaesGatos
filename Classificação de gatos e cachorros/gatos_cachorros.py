from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from tensorflow.keras.preprocessing import image


classificador = Sequential();
classificador.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation="relu"));
classificador.add(BatchNormalization());
classificador.add(MaxPooling2D(pool_size=(2,2)));

classificador.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation="relu"));
classificador.add(BatchNormalization());
classificador.add(MaxPooling2D(pool_size=(2,2)));


#O Flatten é responsável por converter toda matrix da imagem gerada 
#em um vetor que será utilizado pelas camadas da I.A. Ou seja nos passos
#anteriores só estavam sendo extraídas as principais caracteristicas da
#imagem atraves dos Kernels e do MaxPooling.
classificador.add(Flatten());


classificador.add(Dense(units = 128,activation='relu'));
classificador.add(Dropout(0.2));
classificador.add(Dense(units = 128,activation="relu"));
classificador.add(Dropout(0.2));

classificador.add(Dense(units=1,activation = "sigmoid"));

classificador.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy']);


#rescale para escalar as imagens.
#OBS: o mesmo escalonamento utilizado no treinamento é o mesmo
#utilizado no treinamento.
gerador_treinamento = ImageDataGenerator(rescale=1./255,
                                         rotation_range=7,
                                         horizontal_flip=True,
                                         shear_range=0.2,
                                         height_shift_range=0.07,
                                         zoom_range=0.2);

gerador_teste = ImageDataGenerator(rescale=1./255);


#target_size é o mesmo utilizado no input_shape
#para a criação da arquitetura da CNN.
#OBS: o mesmo ocorre no target_size da base de teste.
base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size=(64,64),
                                                           batch_size=32,
                                                           class_mode='binary');

base_teste = gerador_teste.flow_from_directory('dataset/test_set',
                                               target_size=(64,64),
                                               batch_size=32,
                                               class_mode='binary');


#Em steps_per_epoch o mais recomendado é colocar o número
#total de imagens de treinamento pois assim o modelo ao ser treinado ira
#passar por todas as imagems a cada época, porém esse tipo de abordagem
#consome muito processamento, por isso o numero total de imagens está sendo
#dividito por 32 que coicidentemente é o mesmo núemro do batch, porém
#pode ser divido por números menores ou maiores dependendo da necessidade.

#O mesmo está sendo feito no validation_steps que no caso é a quantidade
#de imagens de teste
classificador.fit_generator(base_treinamento,steps_per_epoch=4000/32,
                            epochs=10,
                            validation_data=base_teste,
                            validation_steps=1000/32)


#Carrega a imagem e deixa ela em 64x64
imagem_teste = image.load_img('dataset/test_set/gato/cat.3500.jpg',target_size=(64,64));

#Converte a imagem em um array
imagem_teste =image.img_to_array(imagem_teste)

#Normaliza os valores de 0 até 255
imagem_teste /=255

#Expande as dimensões do array. No caso está criando mais uma coluna
#que informa a quantidade de imagens por array (no caso temos 1 imagem por array)
imagem_teste =np.expand_dims(imagem_teste, axis=0);

previsao = classificador.predict(imagem_teste);

#Retorna os indices referente a cada classe (no caso 0 para cachorro e 1 para gato)
base_treinamento.class_indices