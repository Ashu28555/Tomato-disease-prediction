# tomato-disease-prediction
## AIM: 
The aim of this project is to develop and evaluate convolutional neural network (CNN) models for image classification of tomato leaf diseases using TensorFlow and Keras. The objective is to build models capable of accurately classifying images and providing corresponding precautions and treatments based on the predicted disease.

## OBJECTIVE :
1. Implement CNN architectures using TensorFlow and Keras.
2. Train the models on a dataset of images.
3. Evaluate the models' performance on both training and validation datasets.
4. Optimize the models' architecture and hyperparameters to achieve higher accuracy.
5. Save the trained models for future use or deployment.

## DATASET :
The collection of datasets from Kaggle comprises 
10,000 images of tomato leaves that are available 
to the public and represent 9 illnesses and 1 healthy condition. 
Tomato Diseases are :
1. Bacterial spot
2. Early blight
3. Late blight
4. Leaf Mold
5. Septoria leaf spot
6. Spider mites Two spotted spider mite
7. Target Spot
8. Yellow Leaf Curl Virus
9. Mosaic virus
10. Healthy

### Importing important libraries

import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Resizing, Rescaling
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

### Defin important measures

BATCH_SIZE = 32
IMAGE_SIZE = 224
CHANNELS = 3
EPOCHS = 50

### Load the dataset

data = tf.keras.preprocessing.image_dataset_from_directory(
    'train',
    shuffle = True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size =BATCH_SIZE
)

### Pre- Processing of data

class_names = data.class_names
class_names

label_counts = {}

# Iterate through the dataset to count occurrences of each label
for images_batch, labels_batch in data:
    for label in labels_batch.numpy():
        label_name = class_names[label]  # Get the label name from the class_names list
        if label_name in label_counts:
            label_counts[label_name] += 1
        else:
            label_counts[label_name] = 1

# Print the label counts
print("Label Counts:")
for label, count in label_counts.items():
    print(f"{label}: {count}")


plt.figure(figsize = (12,12))
for image_batch, labels_batch in data.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())
    for i in range(12):
        ax = plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype('uint8'))
        plt.title(class_names[labels_batch[i]])
        plt.axis('off')

### Split the datset into Training, Testing and Validation dataset

def get_dataset_partitions_tf(ds, train_split = 0.8, val_split = 0.1, test_split = 0.1, shuffle = True, shuffle_size = 10000):
    assert (train_split+test_split+val_split) == 1
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed =12)
        
    train_size = int(train_split*ds_size)
    val_size = int(val_split*ds_size)
    
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_tf(data)

print(len(train_ds),
len(test_ds),
len(val_ds))

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)

resize_and_rescale = tf.keras.Sequential([
    Resizing(IMAGE_SIZE,IMAGE_SIZE),
    Rescaling(1.00/255)
])

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
])

input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 10

### Convolution Neural Network Layer

model = Sequential([
    Input(shape=input_shape),
    resize_and_rescale,
    data_augmentation,
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(n_classes, activation='softmax')
])


model.summary()

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= False),
    metrics = ['accuracy']
)

### Training the dataset using CNN 

history = model.fit(
    train_ds,
    batch_size = BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs = EPOCHS
)

After training the datset by using CNN we got the :
1. Training Accuracy = 95.82 %
2. Validation Accuracy = 94.46 %
3. Training Loss = 0.1263
4. Validation Loss = 0.1559

### Testing the dataset

scores = model.evaluate(test_ds)

After evalustion of testing dataset we got :
1. Accuracy = 96.47%
2. Loss = 0.1172 

scores

history.params

history.history.keys()

len(history.history['loss'])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

### 1. Visualization of Training and Validation Accuracy with respect to epochs 
### 2. Visualization of Training and Validation Loss with respect to epochs 

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS), acc, label = 'Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')
            
plt.subplot(1,2,2)
plt.plot(range(EPOCHS), loss, label = 'Training Loss')
plt.plot(range(EPOCHS), val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')           
            

### Testing a image :

for images_batch, labels_batch in test_ds.take(1):
    first_image = images_batch[0].numpy().astype('uint8')
    plt.imshow(first_image)

for images_batch, labels_batch in test_ds.take(1):
    first_image = images_batch[0].numpy().astype('uint8')
    print('First Image to predict')
    plt.imshow(first_image)
    print('Actual Label :',class_names[labels_batch[0].numpy()])
    
    batch_prediction = model.predict(images_batch)
    print('Predicted Label',class_names[np.argmax(batch_prediction[0])])

def predict(model,img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array,0)
    
    predictions = model.predict(img_array)
    
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

plt.figure(figsize = (18,18))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        predicted_class, confidence = predict(model, images[i].numpy())
        
        actual_class = class_names[labels[i]]
        plt.title(f'Actual : {actual_class}, \n Predicted : {predicted_class}. \n Confidence: {confidence}%')
        plt.axis('off')

### Saving the Model

model_version = 1
model.save(f'../models/model_v{model_version}.keras')



## Load the Model

loaded_model = tf.keras.models.load_model('../models/model_v2.keras')

for images_batch, labels_batch in test_ds.take(1):
    first_image = images_batch[0].numpy().astype('uint8')
    print('First Image to predict')
    plt.imshow(first_image)
    print('Actual Label :',class_names[labels_batch[0].numpy()])
    
    batch_prediction = loaded_model.predict(images_batch)
    predicted_label = class_names[np.argmax(batch_prediction[0])]
    plt.text(10, 30, f'Predicted Label: {predicted_label}', color='red', fontsize=12, weight='bold')

### Precautions and treatments of tomato disease with respect to diseases.

disease_info = {
    'Tomato___Bacterial_spot': {
        'precautions': ['Select tomato varieties that are bred to be resistant to bacterial spot if available. Resistant varieties can greatly reduce the likelihood of infection.',
                        'Rotate tomato crops with non-host crops such as beans, corn, or cucurbits to break the disease cycle. Avoid planting tomatoes in the same area for consecutive years.',
                        'Practice good garden hygiene by removing and destroying infected plant debris. This helps reduce the source of bacterial inoculum and prevents the disease from spreading.',
                        'Water tomato plants at the base using drip irrigation or a soaker hose to keep foliage dry. Avoid overhead watering, which can spread bacteria through splashing.',
                        'Apply a layer of organic mulch around tomato plants to prevent soil splash onto lower leaves. Mulch helps maintain soil moisture and suppress weed growth, reducing the risk of infection.'],
        'treatment': ['Regularly inspect tomato plants for symptoms of bacterial spot, such as dark, water-soaked lesions on leaves, stems, and fruit. Remove and destroy infected plant parts promptly to prevent further spread of the disease.',
                      'Apply copper-based bactericides according to label instructions. Copper compounds can help suppress bacterial growth and reduce the severity of bacterial spot.',
                      'Prune and thin tomato plants to improve airflow and reduce humidity around the foliage. This helps create an environment less favorable for bacterial spot development.',
                      'Consider using biological control agents containing beneficial bacteria, such as Bacillus subtilis. These agents can compete with and suppress the growth of pathogenic bacteria.',
                       'Avoid working in the garden when foliage is wet to minimize the risk of spreading fungal spores.Ensure proper nutrition and avoid excessive nitrogen fertilization, as lush growth can be more susceptible to fungal diseases.',
                      'Plant tomato varieties that are known to exhibit resistance or tolerance to early blight if available. Resistant varieties can help reduce the severity of the disease and may require fewer chemical interventions.',
                      'Monitor treated plants regularly for signs of new infections or disease progression. Repeat fungicide applications as needed, following label instructions and recommended intervals.']
    },
    'Tomato___Early_blight': {
        'precautions': ['Rotate tomato crops with non-host crops such as beans, corn, or cucurbits to disrupt the disease cycle. Avoid planting tomatoes in the same location for consecutive years.',
                        'Practice good garden hygiene by removing and destroying infected plant debris. This helps reduce the source of fungal spores and prevents the disease from spreading.',
                        'Water tomato plants at the base using drip irrigation or a soaker hose to keep foliage dry. Avoid overhead watering, which can splash soil-borne fungal spores onto leaves.',
                        'Apply a layer of organic mulch around tomato plants to prevent soil splash onto lower leaves. Mulch helps maintain soil moisture and suppress weed growth, reducing the risk of infection.',
                        'Prune lower leaves of tomato plants to improve airflow and reduce humidity around the foliage. This can help prevent the spread of the disease by creating a less favorable environment for fungal growth.'],
        'treatment': ['Inspect tomato plants regularly and remove any leaves showing symptoms of early blight. Infected leaves should be carefully pruned and disposed of to prevent further spread of the disease.',
                      'Apply fungicides labeled for early blight control according to the manufacturers instructions. Fungicides containing chlorothalonil, mancozeb, or copper are commonly used for early blight management.',
                      'Consider using biological control agents such as Bacillus subtilis-based products, which can help suppress the growth of fungal pathogens. Follow the manufacturers recommendations for application and timing.',
                      'Avoid working in the garden when foliage is wet to minimize the risk of spreading fungal spores.Ensure proper nutrition and avoid excessive nitrogen fertilization, as lush growth can be more susceptible to fungal diseases.',
                      'Plant tomato varieties that are known to exhibit resistance or tolerance to early blight if available. Resistant varieties can help reduce the severity of the disease and may require fewer chemical interventions.',
                      'Monitor treated plants regularly for signs of new infections or disease progression. Repeat fungicide applications as needed, following label instructions and recommended intervals.']
    },
    'Tomato___Late_blight': {
        'precautions': ['Whenever possible, choose tomato varieties that are resistant or tolerant to late blight. Resistant varieties can significantly reduce the likelihood of infection.',
                        'Rotate tomato crops with non-host crops such as beans, corn, or cucurbits to break the disease cycle. Avoid planting tomatoes in the same area for consecutive years.',
                        'Practice good garden hygiene by removing and destroying infected plant debris. This helps reduce the source of fungal spores and prevents the disease from spreading.',
                        'Water tomato plants at the base using drip irrigation or a soaker hose to keep foliage dry. Avoid overhead watering, which can spread fungal spores through splashing.',
                        'Space tomato plants adequately to improve airflow and reduce humidity within the canopy. Prune lower leaves to increase ventilation and reduce the risk of infection.'],
        'treatment': ['Apply fungicides labeled for late blight control according to the manufacturer instructions. Fungicides containing chlorothalonil, mancozeb, or copper are commonly used for late blight management.',
                      'Consider using biological control agents containing beneficial microorganisms, such as Bacillus subtilis. These agents can help suppress the growth of fungal pathogens and reduce disease severity.',
                      'Avoid working in the garden when foliage is wet to minimize the risk of spreading fungal spores.Ensure proper nutrition and avoid excessive nitrogen fertilization, as lush growth can be more susceptible to late blight.',
                      'Plant tomato varieties that are known to exhibit resistance or tolerance to late blight if available. Resistant varieties are less likely to succumb to the disease even under favorable conditions.']
    },
    'Tomato___Leaf_Mold': {
        'precautions': ['Choose tomato varieties that are resistant or tolerant to leaf mold if available. Resistant varieties can significantly reduce the likelihood of infection.',
                        'Rotate tomato crops with non-host crops such as beans, corn, or cucurbits to break the disease cycle. Avoid planting tomatoes in the same area for consecutive years.',
                        'Practice good garden hygiene by removing and destroying infected plant debris. This helps reduce the source of fungal spores and prevents the disease from spreading.',
                        'Space tomato plants adequately to improve airflow and reduce humidity within the canopy. Prune lower leaves to increase ventilation and reduce the risk of infection.',
                        'Water tomato plants at the base using drip irrigation or a soaker hose to keep foliage dry. Avoid overhead watering, which can promote leaf wetness and fungal growth.'],
        'treatment': ['Apply fungicides labeled for leaf mold control according to the manufacturers instructions. Fungicides containing copper or potassium bicarbonate are commonly used for leaf mold management.',
                      'Consider using biological control agents containing beneficial microorganisms, such as Bacillus subtilis. These agents can help suppress the growth of fungal pathogens and reduce disease severity.',
                      'Remove and destroy infected leaves promptly to prevent the spread of the disease.Ensure proper nutrition and avoid excessive nitrogen fertilization, as lush growth can be more susceptible to leaf mold.',
                      'Monitor treated plants regularly for signs of new infections or disease progression. Repeat fungicide applications as needed, following label instructions and recommended intervals.']
    },
    'Tomato___Septoria_leaf_spot': {
        'precautions': ['Choose tomato varieties that are resistant or tolerant to Septoria leaf spot if available. Resistant varieties can significantly reduce the likelihood of infection.',
                        'Rotate tomato crops with non-host crops such as beans, corn, or cucurbits to break the disease cycle. Avoid planting tomatoes in the same area for consecutive years.',
                        'Practice good garden hygiene by removing and destroying infected plant debris. This helps reduce the source of fungal spores and prevents the disease from spreading.',
                        'Space tomato plants adequately to improve airflow and reduce humidity within the canopy. Prune lower leaves to increase ventilation and reduce the risk of infection.',
                        'Water tomato plants at the base using drip irrigation or a soaker hose to keep foliage dry. Avoid overhead watering, which can promote leaf wetness and fungal growth.'],
        'treatment': ['Apply fungicides labeled for Septoria leaf spot control according to the manufacturers instructions. Fungicides containing chlorothalonil, mancozeb, or copper are commonly used for Septoria leaf spot management.',
                      'Consider using biological control agents containing beneficial microorganisms, such as Bacillus subtilis. These agents can help suppress the growth of fungal pathogens and reduce disease severity.',
                      'Plant tomato varieties that are known to exhibit resistance or tolerance to Septoria leaf spot if available. Resistant varieties are less likely to succumb to the disease even under favorable conditions.',
                      'Monitor treated plants regularly for signs of new infections or disease progression. Repeat fungicide applications as needed, following label instructions and recommended intervals.']
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'precautions': ['Encourage natural predators of spider mites, such as ladybugs, lacewings, and predatory mites, by planting companion plants that attract them or by releasing them into the garden.',
                        'Keep the garden area clean by removing weeds, debris, and overgrown vegetation, as these can harbor spider mites and other pests.',
                        'Avoid excessive nitrogen fertilization, as it can promote lush growth that is more susceptible to spider mite infestations.',
                        'Maintain proper humidity levels by watering plants in the morning and providing adequate ventilation to reduce conditions favorable for spider mite reproduction.'],
        'treatment': ['Use a strong stream of water to spray infested tomato plants, focusing on the undersides of leaves where spider mites typically reside. This can help physically remove spider mites and disrupt their feeding.',
                      'Apply insecticidal soap or horticultural oil to affected plants, ensuring thorough coverage of both sides of the leaves. These products suffocate spider mites on contact.',
                      'Use botanical insecticides derived from plants such as neem oil, pyrethrin, or rosemary oil. These can provide effective control of spider mites while being less harmful to beneficial insects.',
                      'Consider using systemic insecticides that are specifically labeled for spider mite control. These products are absorbed by the plant and can kill spider mites when they feed on treated foliage.',
                      'If possible, isolate heavily infested tomato plants to prevent the spread of spider mites to other plants in the garden.']
    },
    'Tomato___Target_Spot': {
        'precautions': ['Choose tomato varieties that are resistant or tolerant to Target Spot if available. Resistant varieties can significantly reduce the likelihood of infection.',
                        'Rotate tomato crops with non-host crops such as beans, corn, or cucurbits to break the disease cycle. Avoid planting tomatoes in the same area for consecutive years.',
                        'Practice good garden hygiene by removing and destroying infected plant debris. This helps reduce the source of fungal spores and prevents the disease from spreading.',
                        'Space tomato plants adequately to improve airflow and reduce humidity within the canopy. Prune lower leaves to increase ventilation and reduce the risk of infection.',
                        'Water tomato plants at the base using drip irrigation or a soaker hose to keep foliage dry. Avoid overhead watering, which can promote leaf wetness and fungal growth.'],
        'treatment': ['Apply fungicides labeled for Target Spot control according to the manufacturers instructions. Fungicides containing chlorothalonil, mancozeb, or copper are commonly used for Target Spot management.',
                      'Consider using biological control agents containing beneficial microorganisms, such as Bacillus subtilis. These agents can help suppress the growth of fungal pathogens and reduce disease severity.',
                      'Plant tomato varieties that are known to exhibit resistance or tolerance to Target Spot if available. Resistant varieties are less likely to succumb to the disease even under favorable conditions.',
                      'Ensure proper nutrition and avoid excessive nitrogen fertilization, as lush growth can be more susceptible to Target Spot.']
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'precautions': ['Plant tomato varieties that are resistant or tolerant to Tomato Yellow Leaf Curl Virus if available. Resistant varieties can significantly reduce the likelihood of infection.',
                        'Implement measures to control the population of silverleaf whiteflies, which transmit TYLCV. This may include using insecticidal soaps, neem oil, reflective mulches, or introducing natural predators such as ladybugs or lacewings.',
                        'Rotate tomato crops with non-host crops to break the disease cycle and reduce the buildup of virus inoculum in the soil.',
                        'Remove weeds, especially those in the Solanaceae family (such as nightshade), which can serve as alternative hosts for the virus and whiteflies.',
                        'Practice good garden hygiene by removing and destroying infected plant debris. This helps reduce the source of virus inoculum and prevents the disease from spreading.'],
        'treatment': ['Monitor tomato plants regularly for symptoms of TYLCV, such as yellowing and curling of leaves, stunted growth, and distorted fruit. Remove and destroy infected plants promptly to prevent the virus from spreading to healthy plants.',
                      'Use insecticides labeled for controlling silverleaf whiteflies if their population becomes problematic. Follow label instructions carefully and avoid broad-spectrum insecticides that may harm beneficial insects.',
                      'Monitor plants closely for symptoms and remove any infected plants immediately.',
                      'Plant tomato varieties that have been bred for resistance to TYLCV. While resistant varieties may still become infected, they often show fewer symptoms and yield better than susceptible varieties under virus pressure.',
                      'Maintain optimal growing conditions for tomatoes, including adequate water, sunlight, and nutrients. Stressed plants are more susceptible to TYLCV infection and may exhibit more severe symptoms.']
    },
    'Tomato___Tomato_mosaic_virus': {
        'precautions': ['Start with virus-free tomato seedlings or transplants from reputable sources to reduce the risk of introducing Tomato Mosaic Virus into your garden.',
                        'Practice good garden hygiene by removing and destroying infected plant debris. This helps reduce the source of virus inoculum and prevents the disease from spreading.',
                        'Manage pests that can transmit ToMV, such as aphids and whiteflies, using insecticidal soaps, neem oil, or other appropriate control methods.',
                        'Avoid working with tomato plants when hands are contaminated with soil or plant debris. Wash hands and disinfect tools between handling plants to prevent the spread of ToMV.',
                        'Remove weeds, especially those in the Solanaceae family (such as nightshade), which can serve as alternative hosts for ToMV.'],
        'treatment': ['Monitor tomato plants regularly for symptoms of Tomato Mosaic Virus, such as mottled or distorted leaves, stunted growth, and reduced yield. Remove and destroy infected plants promptly to prevent the virus from spreading to healthy plants.',
                      'Implement measures to control populations of aphids and whiteflies, which can transmit ToMV. This may include using insecticidal soaps, neem oil, reflective mulches, or introducing natural predators such as ladybugs or lacewings.',
                      'Plant tomato varieties that have been bred for resistance to Tomato Mosaic Virus if available. While resistant varieties may still become infected, they often show fewer symptoms and yield better under virus pressure.',
                      'If possible, isolate tomato plants from other susceptible crops to minimize the risk of virus transmission. Continuously monitor plants for signs of virus infection and take prompt action if symptoms appear.',
                      'Install reflective mulches around tomato plants, as they can deter aphids and whiteflies by reflecting sunlight and disrupting their behavior. This may help reduce virus transmission.']
    },
    'Tomato___healthy': {
        'precautions': ['While your plants are healthy, continue practicing good gardening habits such as proper watering, fertilizing, and soil management to maintain plant vigor and resilience against diseases.',
                        'Regularly monitor plants for signs of disease or pest infestations and take prompt action if any issues arise.'],
        'treatment': []  # No treatment needed for healthy plants
    }
}

for images_batch, labels_batch in test_ds.take(1):
    first_image = images_batch[0].numpy().astype('uint8')
    print('First Image to predict')
    plt.imshow(first_image)
    print('Actual Label:', class_names[labels_batch[0].numpy()])
    
    batch_prediction = loaded_model.predict(images_batch)
    predicted_label_index = np.argmax(batch_prediction[0])
    predicted_label = class_names[predicted_label_index]
    plt.text(10, 30, f'Predicted Label: {predicted_label}', color='red', fontsize=12, weight='bold')
    
    # Check if predicted label corresponds to a disease
    if predicted_label in disease_info:
        print('\nPrecautions:')
        for i, precaution in enumerate(disease_info[predicted_label]['precautions'], start=1):
            print(f'{i}. \033[1m{precaution}\033[0m')
        
        print('\nTreatment:')
        for i, treatment in enumerate(disease_info[predicted_label]['treatment'], start=1):
            print(f'{i}. \033[1m{treatment}\033[0m')
    else:
        print('\nNo precautions or treatment information available for this disease.')


### Conclusion:
In this experiment, we successfully developed and evaluated convolutional neural network (CNN) models for image classification using TensorFlow and Keras. The models were trained on a dataset comprising 10,000 images of tomato leaves representing 10 different classes, including 9 illnesses and 1 healthy condition. The aim was to build models capable of accurately classifying these images into their respective categories.

We implemented CNN architectures consisting of convolutional layers followed by max-pooling layers, flattening, and dense layers for classification. Data preprocessing techniques such as resizing, rescaling, and data augmentation were applied to enhance model performance and generalization.

After training the models for 50 epochs, we achieved promising results:

1. Training Accuracy: 95.82%
2. Validation Accuracy: 94.46%
3. Training Loss: 0.1263
4. Validation Loss: 0.1559

Subsequently, the models were evaluated on a separate testing dataset, yielding an accuracy of 96.47% and a loss of 0.1172.

Visualization of training and validation accuracy, as well as training and validation loss, provided insights into the models' learning progress throughout the training process.

Lastly, we tested the models on sample images from the testing dataset, demonstrating their capability to predict tomato leaf diseases with high accuracy and confidence.

The models were enhanced to provide information on precautions and treatments specific to each predicted disease, offering valuable insights for disease management in tomato plants.

### Learning Outcomes:
1. Image Classification with CNNs: Gain insights into designing CNN architectures for image classification tasks, including preprocessing techniques and model evaluation.
2. Information Integration: Learn to integrate additional information (precautions and treatments) into the prediction pipeline for practical applications.
3. Model Optimization: Explore techniques for optimizing model architecture and hyperparameters to improve classification accuracy and performance.
4. Practical Application: Understand the practical implications of image classification models in agriculture, particularly in plant disease management and crop yield optimization..

