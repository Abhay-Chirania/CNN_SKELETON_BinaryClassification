vfrom keras.models import Sequential
from keras.layers import Dense,Convolution2D,MaxPooling2D,Flatten,Dropout
from keras.preprocessing.image import ImageDataGenerator
def build_classifier():
    classifier=Sequential()
    classifier.add(Convolution2D(32,3,3,input_shape=(128,128,3),activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
    classifier.add(Convolution2D(64,3,3,activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim=64,activation='relu'))
    classifier.add(Dropout(p=0.5))
    classifier.add(Dense(output_dim=1,activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

def train(trainpath,testpath,model_save_path,batch_size=32,epoch=1,totalTrainImage=0,totalTestImage=0):


    train_datagen=ImageDataGenerator(rescale=1./255,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True)
    test_datagen=ImageDataGenerator(rescale=1./255)
    training_set=train_datagen.flow_from_directory(trainpath,
                                                   target_size=(128,128),
                                                   batch_size=batch_size,
                                                   class_mode='binary')
    test_set=test_datagen.flow_from_directory(testpath,
                                              target_size=(128,128),
                                              batch_size=batch_size,
                                              class_mode='binary')
    classifier=build_classifier()

    print("Training Started....")
    classifier.fit_generator(training_set,
                             steps_per_epoch=totalTrainImage/batch_size,
                             epochs=epoch,
                             validation_data=test_set,
                             validation_steps=totalTestImage/batch_size,
                             )
    classifier.save(model_save_path+"TrainedModel.h5")
    print("Model Trained and Saved")


                                    #####MAIN#####

if __name__=="__main__":
    trainpath=""        #full path for training dataset
    testpath=""         #full path for testing dataset
    model_save_path=""  #full path where model needs to be saved after training
    batch_size=0        #batch size of training
    epoch=0             #epochs of training
    totalTrainImage=0   #no of training images
    totalTestImage=0   #no of test images
    #calling training function
    train(trainpath,testpath,model_save_path,batch_size=batch_size,epoch=epoch,totalTrainImage=totalTrainImage,totalTestImage=totalTestImage)     
                         
                                               
