
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB format
        image_fromarray = Image.fromarray(img)
        resize_image = image_fromarray.resize((height, width))  
        image = np.array(resize_image)[np.newaxis, :]
        image = image/255
        # Predict the class using the Autoencoder model
        auto_output = auto.predict(image)
        auto_class = np.argmax(auto_output)
        
        # Create the adversarial pattern and preprocess the attacked image
        j = tf.reshape(auto_output ,(1, 43))
        attacked_img = create_adversarial_pattern(tf.convert_to_tensor(image), j, 0.2)
        image_fromarray1= Image.fromarray(attacked_img[0], 'RGB')
        resize_image = image_fromarray1.resize((height, width))
            
        # Preprocess the image and predict the class using Lenet and Autoencoder models
        attacked_img = np.array(resize_image)[np.newaxis, :]
        attacked_img= image/255

      
        
        # Predict the class using the Autoencoder model on the attacked image
        attacked_output = auto.predict(attacked_img)
        attacked_class = np.argmax(attacked_output)
        