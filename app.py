from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from keras.models import load_model
import base64
import numpy as np 
import cv2,os
from PIL import Image
import io
from PIL import Image
import matplotlib.pyplot as plt


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
np.random.seed(42)
tf.random.set_seed(42)

height = 32
width = 32
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    return render_template('upload.html')

from keras.optimizers import Adam
lenet= load_model('lenet.h5', compile=False)
lenet.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

auto = load_model('auto.h5')

auto_class_names =  { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }
# Define the function to create the adversarial pattern
loss_object = tf.keras.losses.CategoricalCrossentropy()

def Acreate_adversarial_pattern(image, label, e=0.53):
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = auto(image)
        loss = loss_object(label, prediction)
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    adversary = (image + (signed_grad * e))
    
    # Clip the pixel values to be within the [0,1] range
    adversary = tf.clip_by_value(adversary, clip_value_min=0.0, clip_value_max=1.0)
    
    # Ensure that the shape of the adversary is the same as the input image
    assert image.shape == adversary.shape, "Adversarial image shape mismatch"
    
    return adversary

def Lcreate_adversarial_pattern(image, label, e=0.53):
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = lenet(image)
        loss = loss_object(label, prediction)
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    adversary = (image + (signed_grad * e))
    
    # Clip the pixel values to be within the [0,1] range
    adversary = tf.clip_by_value(adversary, clip_value_min=0.0, clip_value_max=1.0)
    
    # Ensure that the shape of the adversary is the same as the input image
    assert image.shape == adversary.shape, "Adversarial image shape mismatch"
    
    return adversary

# Define the route for the web page
@app.route('/', methods=['GET', 'POST'])
def getprediction():
    if request.method == 'POST':
        file = request.files.get('image')
        if file is None:
            return "No file uploaded"
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        img = cv2.imread(filename)
        image_fromarray = Image.fromarray(img, 'RGB')
        resize_image = image_fromarray.resize((height, width))  
        image = np.array(resize_image)[np.newaxis, :]
        image = image/255
        # Predict the class using the Autoencoder model
        auto_output = auto.predict(image)
        auto_class = np.argmax(auto_output)
        # Predict the class using the lenet model
        lenet_output = lenet.predict(image)
        lenet_class = np.argmax(lenet_output)
        
        # Create the adversarial pattern and preprocess the attacked image
        attackedimage =cv2.imread(filename)
        image_fromarray = Image.fromarray(attackedimage, 'RGB')
        resize_image = image_fromarray.resize((height, width))
        attackedimage = np.array(resize_image)[np.newaxis, :]
        attackedimage= attackedimage/255
        j = tf.reshape(auto_output ,(1, 43))

        Lattackedimage= Lcreate_adversarial_pattern(tf.convert_to_tensor(attackedimage), j, 0.2)
        lenet_outputs=lenet.predict(Lattackedimage)
        Lattacked_class=np.argmax(lenet_outputs)
        Lattackedimage = Lattackedimage.numpy()
        Lattackedimage = Lattackedimage[0]
        Lattackedimage = np.clip(Lattackedimage, 0.0, 1.0)  # Clip pixel values to be within [0,1]
        
        Aattackedimage= Acreate_adversarial_pattern(tf.convert_to_tensor(attackedimage), j, 0.2)
        auto_outputs = auto.predict(Aattackedimage)
        Aattacked_class = np.argmax(auto_outputs)        
        Aattackedimage = Aattackedimage.numpy()
        Aattackedimage = Aattackedimage[0]
        Aattackedimage = np.clip(Aattackedimage, 0.0, 1.0)  # Clip pixel values to be within [0,1]
        
        # create a figure object using Matplotlib
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        
        # plot the input and attacked images
        input_img_pil = Image.fromarray(img)
        ax[0].imshow(input_img_pil)
        ax[0].set_title(f"Input image:\n {auto_class_names[auto_class]}")
        ax[0].axis('off')
        
        # input_img_pil = Image.fromarray(img)
        # ax[1].imshow(input_img_pil)
        # ax[1].set_title(f" {auto_class_names[lenet_class]}")
        # ax[1].axis('off')
        attacked_img_pil = Image.fromarray((Lattackedimage*255).astype(np.uint8), 'RGB')
        ax[1].imshow(attacked_img_pil)
        ax[1].set_title(f" Attacked image:\n {auto_class_names[Lattacked_class]}")
        ax[1].axis('off')
        
        attacked_img_pil = Image.fromarray((Aattackedimage*255).astype(np.uint8), 'RGB')
        ax[2].imshow(attacked_img_pil)
        ax[2].set_title(f" Defended image: \n{auto_class_names[Aattacked_class]}")
        ax[2].axis('off')

        # save the plot as a PNG image in memory
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes,  dpi=100,format='png')
        img_bytes.seek(0)
        
        # convert the PNG image bytes to a base64-encoded string
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
        
        # render the output HTML page with the plot and the predicted classes
        return render_template('output.html', img_base64=img_base64,auto=auto_class_names[auto_class], auto_class=auto_class_names[Aattacked_class],lenet_class=auto_class_names[Lattacked_class])
    
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)




    